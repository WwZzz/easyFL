import datetime
import errno
import os
import time
from collections import defaultdict, deque
import flgo.benchmark.toolkits.cv.detection.presets as presets
import torch
import torchvision
import torch.distributed as dist
import collections
import numpy as np
from sklearn.metrics import auc

def evaluate(model, dataloader, num_classes, device):
    model.eval()
    metric_dict_by_class = [collections.defaultdict(list) for _ in range(num_classes)]
    for batch_id, (images, targets) in enumerate(dataloader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        output = model(images)

        for label in range(num_classes):
            precision, recall, ap, num_class_obj = precision_recall_ap(targets, output, iou_thresh=0.5, class_label=label)
            if num_class_obj > 0:
                mean_ap = mean_average_precision(targets, output, class_label=label)
                acc = accuracy(targets, output, class_label=label)
                red = redundancy(targets, output, class_label=label)
                metric_dict_by_class[label]['num_objs'].append(num_class_obj)
                metric_dict_by_class[label]['precision'].append(precision)
                metric_dict_by_class[label]['recall'].append(recall)
                metric_dict_by_class[label]['ap'].append(ap)
                metric_dict_by_class[label]['mAP'].append(mean_ap)
                metric_dict_by_class[label]['accuracy'].append(acc)
                metric_dict_by_class[label]['redundancy'].append(red)
    mkeys = ['precision', 'recall', 'mAP', 'AP', 'accuracy', 'redundancy']
    res = {mk: [] for mk in mkeys}
    res['ap_class'] = []
    for label in range(num_classes):
        if len(metric_dict_by_class[label]['num_objs']) == 0:
            continue
        num_class_objs = sum(metric_dict_by_class[label]['num_objs'])
        for mk in mkeys:
            res[mk].append(sum(ni * pi for ni, pi in zip(metric_dict_by_class[label]['num_objs'],
                                                         metric_dict_by_class[label][mk])) / num_class_objs)
        res['ap_class'].append(label)
    return res

def iou(real_box, pred_box):
    """Calculates Intersection over Union for real and predicted bounding box."""

    if len(real_box) == 0:
        raise "Length of target box is 0."
    if len(pred_box) == 0:
        raise "Length of predicted box is 0."
    x1 = max(real_box[0], pred_box[0])
    y1 = max(real_box[1], pred_box[1])
    x2 = min(real_box[2], pred_box[2])
    y2 = min(real_box[3], pred_box[3])
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    real_box_area = (real_box[2] - real_box[0] + 1) * (real_box[3] - real_box[1] + 1)
    pred_box_area = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
    return intersection_area / (real_box_area + pred_box_area - intersection_area)

def average_precision(rec, prec):
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1 + i] != mrec[i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

def precision_recall_ap(targets, predictions, iou_thresh=0.5, class_label=None):
    """Calculates Precision, Recall and Average Precision for given threshold."""
    tp_list = []
    num_gt_boxes = 0
    for target in targets:
        for idx, box in enumerate(target['boxes']):
            if target['labels'][idx] == class_label or class_label is None:
                num_gt_boxes += 1
    for target, pred in zip(targets, predictions):
        if len(pred['boxes']) == 0:
            continue
        if len(target['boxes']) == 0:
            for i in range(len(pred['boxes'])):
                tp_list.append(0)
        for target_idx, target_box in enumerate(target['boxes']):
            if target['labels'][target_idx] != class_label and class_label is not None:
                continue
            is_append = False
            for pred_box in pred['boxes']:
                if iou(target_box, pred_box) > iou_thresh:
                    tp_list.append(1)
                    is_append = True
                    break
            tp_list.append(0) if not is_append else None
    if len(tp_list) == 0:
        if num_gt_boxes == 0:
            return 1, 1, 1, num_gt_boxes
        return 0, 0, 0, num_gt_boxes
    if num_gt_boxes == 0:
        return 0, 0, 0, num_gt_boxes
    tp_list.sort(reverse=True)
    precision_values = [sum(tp_list[:i+1])/len(tp_list[:i+1]) for i in range(len(tp_list))]
    recall_values = [sum(tp_list[:i+1])/num_gt_boxes for i in range(len(tp_list))]
    precision, recall = precision_values[-1], recall_values[-1]
    average_precision = precision if len(precision_values) == 1 else auc(recall_values, precision_values)
    return precision, recall, average_precision, num_gt_boxes

def f1_score(precision, recall):
    """Calculates F1-score."""

    return 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)

def mean_average_precision(targets, predictions, class_label=None, thresh_values=np.arange(0.5, 1, 0.05)):
    """Calculates mean Average Precision."""

    ap_list = []
    for thresh in thresh_values:
        ap_list.append(precision_recall_ap(targets, predictions, thresh, class_label)[2])
    return np.mean(ap_list)

def accuracy(targets, predictions, iou_thresh=0.5, class_label=None):
    """Calculates batch accuracy."""

    num_correct_predictions = 0
    num_predictions = 0
    for pred in predictions:
        for idx, box in enumerate(pred['boxes']):
            if pred['labels'][idx] == class_label or class_label is None:
                num_predictions += 1
    for target, pred in zip(targets, predictions):
        if len(target['boxes']) == 0 or len(pred['boxes']) == 0:
            continue
        for target_idx, target_box in enumerate(target['boxes']):
            if target['labels'][target_idx] != class_label and class_label is not None:
                continue
            for pred_idx, pred_box in enumerate(pred['boxes']):
                if iou(target_box, pred_box) > iou_thresh \
                        and target['labels'][target_idx] == pred['labels'][pred_idx]:
                    num_correct_predictions += 1
                    break
    return 1 if num_predictions == 0 else num_correct_predictions / num_predictions

def redundancy(targets, predictions, class_label=None):
    """Calculates redundancy of predictions."""
    num_predictions, num_gt_boxes = 0, 0
    for pred in predictions:
        for idx, box in enumerate(pred['boxes']):
            if pred['labels'][idx] == class_label or class_label is None:
                num_predictions += 1
    for target in targets:
        for idx, box in enumerate(target['boxes']):
            if target['labels'][idx] == class_label or class_label is None:
                num_gt_boxes += 1
    return num_predictions if num_gt_boxes == 0 else num_predictions / num_gt_boxes

def get_transform(train, data_augmentation:str='hflip'):
    if train:
        return presets.DetectionPresetTrain(data_augmentation=data_augmentation)
    else:
        return presets.DetectionPresetEval()

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def collate_fn(batch):
    return tuple(zip(*batch))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)