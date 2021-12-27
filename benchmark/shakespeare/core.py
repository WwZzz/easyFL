from benchmark.toolkits import download_from_url, extract_from_zip
from benchmark.toolkits import DefaultTaskGen, TupleDataset
import collections
import re
import os
from benchmark.toolkits import  XYTaskReader, ClassifyCalculator
import numpy as np
import os.path
import ujson

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients=1, skewness=0.5, minvol = 10):
        super(TaskGen, self).__init__(benchmark='shakespeare',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/shakespeare/data',
                                      minvol=minvol
                                      )
        # Regular expression to capture an actors name, and line continuation
        self.CHARACTER_RE = re.compile(r'^  ([a-zA-Z][a-zA-Z ]*)\. (.*)')
        self.CONT_RE = re.compile(r'^    (.*)')
        # The Comedy of Errors has errors in its indentation so we need to use
        # different regular expressions.
        self.COE_CHARACTER_RE = re.compile(r'^([a-zA-Z][a-zA-Z ]*)\. (.*)')
        self.COE_CONT_RE = re.compile(r'^(.*)')
        self.ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
        self.NUM_LETTERS = len(self.ALL_LETTERS)
        self.SEQ_LENGTH = 80
        self.save_data = self.XYData_to_json

    def load_data(self):
        # download, read the raw dataset and store it as .json
        raw_path = os.path.join(self.rawdata_path, 'raw_data')
        all_data_path = os.path.join(raw_path, 'all_data.json')
        if not os.path.exists(all_data_path):
            if not os.path.exists(raw_path):
                os.mkdir(raw_path)
            src_path = download_from_url("http://www.gutenberg.org/files/100/old/1994-01-100.zip",
                                              os.path.join(raw_path, 'tmp'))
            tar_paths = extract_from_zip(src_path, raw_path)
            os.remove(src_path)
            with open(tar_paths[0], 'r') as input_file:
                shakespeare_full = input_file.read()
            plays, discarded_lines = self._split_into_plays(shakespeare_full)
            users_and_plays, all_examples, num_skipped = self._get_examples_by_character(plays)
            all_data = {}
            for user in all_examples.keys():
                all_data[user] = {
                    'play': users_and_plays[user],
                    'sound_bites': all_examples[user],
                }
            with open(all_data_path, 'w') as f:
                ujson.dump(all_data, f)
            os.remove(tar_paths[0])
        else:
            with open(os.path.join(self.rawdata_path, 'raw_data', 'all_data.json'), 'r') as f:
                all_data = ujson.load(f)
        # preprocess rawdata
        user_dict = {}
        all_users = [user for user in all_data.keys()]

        train_users = []
        trainXs = []
        trainYs = []
        client_ids = []
        cid = 0
        while (len(train_users) < self.num_clients):
            user = np.random.choice(all_users)
            if user in train_users: continue
            examples = all_data[user]['sound_bites']
            play = all_data[user]['play']
            trainX, trainY = self.example_to_text(examples)
            trainX = self.X_text_to_vec(trainX)
            trainY = self.Y_text_to_vec(trainY)
            l = len(examples)
            if l-max(int(l * 0.2), 1) < self.minvol:
                continue
            else:
                train_users.append(user)
                client_ids.extend([cid for _ in range(l)])
                cid+=1
                trainXs.extend(trainX)
                trainYs.extend(trainY)
        self.train_data = TupleDataset(trainXs, client_ids, trainYs)
        test_users = []
        all_users = list(set(all_users) - set(train_users))
        num_test = min(max(int(0.2 * len(train_users)), 10), len(all_users))
        test_dict = {'x':[], 'y':[]}
        while (len(test_users) < num_test and all_users):
            user = np.random.choice(all_users)
            if user in test_users: continue
            test_users.append(user)
            test_examples = all_data[user]['sound_bites']
            play = all_data[user]['play']
            testX, testY = self.example_to_text(test_examples)
            test_dict['x'].extend(self.X_text_to_vec(testX))
            test_dict['y'].extend(self.Y_text_to_vec(testY))
        self.test_data = test_dict
        return

    def convert_data_for_saving(self):
        xs, _, ys = self.train_data.tolist()
        self.train_data = {
            'x': xs,
            'y': ys
        }

    def _split_into_plays(self, shakespeare_full):
        """Splits the full data by play."""
        # List of tuples (play_name, dict from character to list of lines)
        plays = []
        discarded_lines = []  # Track discarded lines.
        slines = shakespeare_full.splitlines(True)[1:]
        # skip contents, the sonnets, and all's well that ends well
        author_count = 0
        start_i = 0
        for i, l in enumerate(slines):
            if 'by William Shakespeare' in l:
                author_count += 1
            if author_count == 1:
                start_i = i - 5
                break
        slines = slines[start_i:]
        current_character = None
        comedy_of_errors = False
        for i, line in enumerate(slines):
            # This marks the end of the plays in the file.
            if i > 124195 - start_i:
                break
            # This is a pretty good heuristic for detecting the start of a new play:
            if 'by William Shakespeare' in line:
                current_character = None
                characters = collections.defaultdict(list)
                # The title will be 2, 3, 4, 5, 6, or 7 lines above "by William Shakespeare".
                for j in range(2,8):
                    if slines[i - j].strip():
                        title = slines[i-j]
                        break
                title = title.strip()

                assert title, (
                        'Parsing error on line %d. Expecting title 2 or 3 lines above.' %
                        i)
                comedy_of_errors = (title == 'THE COMEDY OF ERRORS')
                # Degenerate plays are removed at the end of the algorithm.
                plays.append((title, characters))
                continue
            # match_character_regex
            match = (self.COE_CHARACTER_RE.match(line) if comedy_of_errors else self.CHARACTER_RE.match(line))
            if match:
                character, snippet = match.group(1), match.group(2)
                # Some character names are written with multiple casings, e.g., SIR_Toby
                # and SIR_TOBY. To normalize the character names, we uppercase each name.
                # Note that this was not done in the original preprocessing and is a
                # recent fix.
                character = character.upper()
                if not (comedy_of_errors and character.startswith('ACT ')):
                    characters[character].append(snippet)
                    current_character = character
                    continue
                else:
                    current_character = None
                    continue
            elif current_character:
                # _match_continuation_regex
                match = (self.COE_CONT_RE.match(line) if comedy_of_errors else self.CONT_RE.match(line))
                if match:
                    if comedy_of_errors and match.group(1).startswith('<'):
                        current_character = None
                        continue
                    else:
                        characters[current_character].append(match.group(1))
                        continue
            # Didn't consume the line.
            line = line.strip()
            if line and i > 2646:
                # Before 2646 are the sonnets, which we expect to discard.
                discarded_lines.append('%d:%s' % (i, line))
        # Remove degenerate "plays".
        return [play for play in plays if len(play[1]) > 1], discarded_lines

    def play_and_character(self, play, character):
        return re.sub('\\W+', '_',(play + '_' + character).replace(' ', '_'))

    def _get_examples_by_character(self, plays):
        skipped_characters = 0
        all_examples = collections.defaultdict(list)
        def add_examples(example_dict, example_tuple_list):
            for play, character, sound_bite in example_tuple_list:
                example_dict[self.play_and_character(
                    play, character)].append(sound_bite)
        users_and_plays = {}
        for play, characters in plays:
            curr_characters = list(characters.keys())
            for c in curr_characters:
                users_and_plays[self.play_and_character(play, c)] = play
            for character, sound_bites in characters.items():
                examples = [(play, character, sound_bite) for sound_bite in sound_bites]
                if len(examples) < 2:
                    skipped_characters += 1
                    # Skip characters with fewer than 2 lines since we need at least one
                    # train and one test line.
                    continue
                add_examples(all_examples, examples)
        return users_and_plays, all_examples, skipped_characters

    def example_to_text(self, examples):
        text = ' '.join(examples)
        text = re.sub(r"   *", r' ', text)
        X = []
        Y = []
        for i in range(0, len(text) - self.SEQ_LENGTH, 1):
            seq_in = text[i:i + self.SEQ_LENGTH]
            seq_out = text[i + self.SEQ_LENGTH]
            X.append(seq_in)
            Y.append(seq_out)
        return X, Y

    def X_text_to_vec(self, X):
        return [[self.ALL_LETTERS.find(c) for c in word] for word in X]

    def Y_text_to_vec(self, Y):
        return [self.ALL_LETTERS.find(c) for c in Y]

class TaskReader(XYTaskReader):
    def __init__(self, taskpath=''):
        super(TaskReader, self).__init__(taskpath)

class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)



