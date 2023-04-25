import urllib
import zipfile
import torch
from torch.utils.data import TensorDataset, Dataset
from flgo.benchmark.toolkits import BasicTaskGenerator, BasicTaskCalculator
from flgo.benchmark.base import BasicTaskPipe
import collections
import re
import os
import os.path
import json
import flgo.benchmark
import os.path
def download_from_url(url= None, filepath = '.'):
    """Download dataset from url to filepath."""
    if url:urllib.request.urlretrieve(url, filepath)
    return filepath

def extract_from_zip(src_path, target_path):
    """Unzip the .zip file (src_path) to target_path"""
    f = zipfile.ZipFile(src_path)
    f.extractall(target_path)
    targets = f.namelist()
    f.close()
    return [os.path.join(target_path, tar) for tar in targets]

class SHAKESPEARE(Dataset):
    def __init__(self, root, train=True):

        self.train = train
        self.root = root
        self.CHARACTER_RE = re.compile(r'^  ([a-zA-Z][a-zA-Z ]*)\. (.*)')
        self.CONT_RE = re.compile(r'^    (.*)')
        # The Comedy of Errors has errors in its indentation so we need to use
        # different regular expressions.
        self.COE_CHARACTER_RE = re.compile(r'^([a-zA-Z][a-zA-Z ]*)\. (.*)')
        self.COE_CONT_RE = re.compile(r'^(.*)')
        self.ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
        self.NUM_LETTERS = len(self.ALL_LETTERS)
        self.SEQ_LENGTH = 80

        file = 'train_data.json' if self.train else 'test_data.json'
        if not os.path.exists(os.path.join(self.root, 'raw_data', file)):
            self.download()
        with open(os.path.join(self.root, 'raw_data', file), 'r') as f:
            data = json.load(f)
            if self.train:
                self.id = data['id']
            self.x = data['x']
            self.y = data['y']

    def __getitem__(self, index):
        return torch.tensor(self.x[index], dtype=torch.long), self.y[index]

    def __len__(self):
        return len(self.x)

    def download(self):
        raw_path = os.path.join(self.root, 'raw_data')
        if not os.path.exists(os.path.join(raw_path, 'all_data.json')):
            if not os.path.exists(raw_path):
                os.makedirs(raw_path)
            print('Downloading...')
            src_path = download_from_url("https://www.gutenberg.org/files/100/old/1994-01-100.zip",
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
            with open(os.path.join(self.root, 'raw_data', 'all_data.json'), 'w') as f:
                json.dump(all_data, f)
            os.remove(tar_paths[0])
        else:
            with open(os.path.join(self.root, 'raw_data', 'all_data.json'), 'r') as f:
                all_data = json.load(f)
        print('Processing...')
        self.process(all_data)

    def process(self, all_data):
        users = list(all_data.keys())
        num_user = len(users)
        s1 = int(num_user * 0.9)
        s2 = num_user - s1
        train_users, test_users = torch.utils.data.random_split(users, [s1, s2])
        train_sample_ids = []
        trainXs = []
        trainYs = []
        testXs = []
        testYs = []
        cid = 0
        for uid in train_users.indices:
            user = users[uid]
            examples = all_data[user]['sound_bites']
            X, Y = self.example_to_text(examples)
            X = self.X_text_to_vec(X)
            Y = self.Y_text_to_vec(Y)
            if len(X) < 10:
                testXs.extend(X)
                testYs.extend(Y)
                continue
            trainXs.extend(X)
            trainYs.extend(Y)
            train_sample_ids.extend([cid] * len(X))
            cid += 1
        train_data = {
            'x': trainXs,
            'y': trainYs,
            'id': train_sample_ids
        }
        with open(os.path.join(self.root, 'raw_data', 'train_data.json'), 'w') as f:
            json.dump(train_data, f)
        for uid in test_users.indices:
            user = list(all_data.keys())[uid]
            examples = all_data[user]['sound_bites']
            X, Y = self.example_to_text(examples)
            X = self.X_text_to_vec(X)
            Y = self.Y_text_to_vec(Y)
            testXs.extend(X)
            testYs.extend(Y)
        test_data = {
            'x': testXs,
            'y': testYs
        }
        with open(os.path.join(self.root, 'raw_data', 'test_data.json'), 'w') as f:
            json.dump(test_data, f)

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


class TaskGenerator(BasicTaskGenerator):
    def __init__(self, rawdata_path=os.path.join(flgo.benchmark.path,'RAW_DATA', 'SHAKESPEARE')):
        super(TaskGenerator, self).__init__(benchmark='shakespeare_classification', rawdata_path=rawdata_path)
        # Regular expression to capture an actors name, and line continuation

    def load_data(self):
        self.train_data = SHAKESPEARE(self.rawdata_path, train=True)
        self.test_data = SHAKESPEARE(self.rawdata_path, train=False)
        return

    def partition(self):
        self.local_datas = self.partitioner(self.train_data)

class TaskPipe(BasicTaskPipe):
    class TaskDataset(torch.utils.data.Subset):
        def __init__(self, dataset, indices):
            super().__init__(dataset, indices)
            self.dataset = dataset
            self.indices = indices

        def __getitem__(self, idx):
            if isinstance(idx, list):
                return self.dataset[[self.indices[i] for i in idx]]
            return self.dataset[self.indices[idx]]

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names, 'server_data': list(range(len(generator.test_data))),
                   'rawdata_path': generator.rawdata_path}
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid], }
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        # load the datasets
        train_data = SHAKESPEARE(root=self.feddata['rawdata_path'], train=True)
        test_data = SHAKESPEARE(root=self.feddata['rawdata_path'], train=False)
        # rearrange data for server
        server_data_test, server_data_valid = self.split_dataset(test_data, running_time_option['test_holdout'])
        task_data = {'server': {'test': server_data_test, 'valid': server_data_valid}}
        # rearrange data for clients
        for cid, cname in enumerate(self.feddata['client_names']):
            cdata = self.TaskDataset(train_data, self.feddata[cname]['data'])
            cdata_train, cdata_valid = self.split_dataset(cdata, running_time_option['train_holdout'])
            task_data[cname] = {'train': cdata_train, 'valid': cdata_valid}
        return task_data

class TaskCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.DataLoader = torch.utils.data.DataLoader

    def compute_loss(self, model, data):
        """
        Args: model: the model to train
                 data: the training dataset
        Returns: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        tdata = self.to_device(data)
        outputs = model(tdata[0])
        loss = self.criterion(outputs, tdata[-1])
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, dataset, batch_size=64, num_workers=0, pin_memory=False):
        """
        Metric = [mean_accuracy, mean_loss]
        Args:
            dataset:
                 batch_size:
        Returns: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        total_loss = 0.0
        num_correct = 0
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.to_device(batch_data)
            outputs = model(batch_data[0])
            batch_mean_loss = self.criterion(outputs, batch_data[-1]).item()
            y_pred = outputs.data.max(1, keepdim=True)[1]
            correct = y_pred.eq(batch_data[-1].data.view_as(y_pred)).long().cpu().sum()
            num_correct += correct.item()
            total_loss += batch_mean_loss * len(batch_data[-1])
        return {'accuracy': 1.0*num_correct/len(dataset), 'loss':total_loss/len(dataset)}

    def to_device(self, data):
        return data[0].to(self.device), data[1].to(self.device)

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return self.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)