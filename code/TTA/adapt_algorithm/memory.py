import random
import torch
import numpy as np


class FIFO:
    def __init__(self, capacity):
        self.data = [[], [], []]
        self.capacity = capacity

    def get_memory(self):
        #print('get memory')
        return self.data

    def get_occupancy(self):
        #print('get occupation')
        return len(self.data[0])

    def add_instance(self, instance):
        #print('add instance')
        if self.get_occupancy() >= self.capacity:
            self.remove_instance()
        for i, dim in enumerate(self.data):
            dim.append(instance[i])

    def remove_instance(self):
        #print('remove instance')
        for dim in self.data:
            dim.pop(0)


class Reservoir:
    def __init__(self, capacity):
        self.data = [[], [], []]
        self.capacity = capacity
        self.counter = 0

    def get_memory(self):
        return self.data

    def get_occupancy(self):
        return len(self.data[0])

    def add_instance(self, instance):
        assert len(instance) == 3
        is_add = True
        self.counter += 1

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance()

        if is_add:
            for i, dim in enumerate(self.data):
                dim.append(instance[i])

    def remove_instance(self):
        m = self.get_occupancy()
        n = self.counter
        u = random.uniform(0, 1)
        if u <= m / n:
            tgt_idx = random.randrange(0, m)
            for dim in self.data:
                dim.pop(tgt_idx)
        else:
            return False
        return True


class PBRS:
    def __init__(self, capacity, num_class):
        self.data = [[[], [], [], [], []] for _ in range(num_class)]  # feat, pseudo_cls, domain, cls, loss
        self.counter = [0] * num_class
        self.marker = [''] * num_class
        self.capacity = capacity
        self.num_class = num_class

    def print_class_dist(self):
        print(self.get_occupancy_per_class())

    def print_real_class_dist(self):
        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            for cls in data_per_cls[3]:
                occupancy_per_class[cls] += 1
        print(occupancy_per_class)

    def get_memory(self):
        tmp_data = [[], [], []]
        for data_per_cls in self.data:
            feats, cls, dls = data_per_cls[0], data_per_cls[1], data_per_cls[2]
            tmp_data[0].extend(feats)
            tmp_data[1].extend(cls)
            tmp_data[2].extend(dls)
        return tmp_data

    def get_occupancy(self):
        return sum(len(data_per_cls[0]) for data_per_cls in self.data)

    def get_occupancy_per_class(self):
        return [len(data_per_cls[0]) for data_per_cls in self.data]

    def update_loss(self, loss_list):
        for data_per_cls in self.data:
            losses = data_per_cls[4]
            for i in range(len(losses)):
                losses[i] = loss_list.pop(0)

    def add_instance(self, instance):
        assert len(instance) == 5
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self):
        max_value = max(self.get_occupancy_per_class())
        return [i for i, oc in enumerate(self.get_occupancy_per_class()) if oc == max_value]

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices:
            largest = random.choice(largest_indices)
            tgt_idx = random.randrange(0, len(self.data[largest][0]))
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:
            m_c = self.get_occupancy_per_class()[cls]
            n_c = self.counter[cls]
            u = random.uniform(0, 1)
            if u <= m_c / n_c:
                tgt_idx = random.randrange(0, len(self.data[cls][0]))
                for dim in self.data[cls]:
                    dim.pop(tgt_idx)
            else:
                return False
        return True
    

import random
import numpy as np

class HUS:
    def __init__(self, capacity, num_class, threshold=None):
        self.data = [[[], [], [], []] for _ in range(num_class)]  # feat, pseudo_cls, domain, conf
        self.counter = [0] * num_class
        self.marker = [''] * num_class
        self.capacity = capacity
        self.threshold = threshold
        self.num_class = num_class

    def set_memory(self, state_dict):  # for tta_attack
        self.data = [[l[:] for l in ls] for ls in state_dict['data']]
        self.counter = state_dict['counter'][:]
        self.marker = state_dict['marker'][:]
        self.capacity = state_dict['capacity']
        self.threshold = state_dict['threshold']
        self.num_class = len(self.data)

    def save_state_dict(self):
        return {
            'data': [[l[:] for l in ls] for ls in self.data],
            'counter': self.counter[:],
            'marker': self.marker[:],
            'capacity': self.capacity,
            'threshold': self.threshold
        }

    def print_class_dist(self):
        print(self.get_occupancy_per_class())

    def print_real_class_dist(self):
        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            for cls in data_per_cls[3]:  # real labels
                occupancy_per_class[cls] += 1
        print(occupancy_per_class)

    def get_memory(self):
        feats_all, cls_all, dls_all = [], [], []
        for data_per_cls in self.data:
            feats_all.extend(data_per_cls[0])
            cls_all.extend(data_per_cls[1])
            dls_all.extend(data_per_cls[2])
        return [feats_all, cls_all, dls_all]

    def get_occupancy(self):
        return sum(len(data_per_cls[0]) for data_per_cls in self.data)

    def get_occupancy_per_class(self):
        return [len(data_per_cls[0]) for data_per_cls in self.data]

    def add_instance(self, instance):
        """
        instance: [feat, pseudo_cls, domain, confidence]
        """
        # 确保输入实例包含4个元素：特征、伪标签、域信息、置信度
        assert len(instance) == 4
        # 提取实例的伪标签，用于后续归类存储
        cls = instance[1]
        # 增加当前伪标签类别遇到的实例计数
        self.counter[cls] += 1
        # 初始化添加标志，默认为True（允许添加）
        is_add = True

        # 若设置了置信度阈值且当前实例置信度低于阈值，则拒绝添加
        if self.threshold is not None and instance[3] < self.threshold:
            is_add = False
        # 若内存容量已满，则尝试移除实例以腾出空间，移除结果决定是否允许添加新实例
        elif self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        # 当允许添加时，将实例各分量存入对应类别的内存列表中
        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self):
        occupancy = self.get_occupancy_per_class()
        max_value = max(occupancy)
        return [i for i, v in enumerate(occupancy) if v == max_value]

    def get_average_confidence(self):
        conf_list = [conf for data_per_cls in self.data for conf in data_per_cls[3]]
        return np.average(conf_list) if conf_list else 0

    def get_target_index(self, data):
        return random.randrange(len(data))

    def remove_instance(self, cls):
        """
        从内存中移除实例，优先从占用量最大的类别中移除，以维持内存容量平衡
        
        Args:
            cls: 当前待添加实例的伪标签类别
        Returns:
            bool: 移除操作是否成功（此处恒为True，因移除逻辑确保至少会移除一个实例）
        """
        # 获取当前占用量最大的类别索引列表（可能存在多个类别占用量相同且最大）
        largest_indices = self.get_largest_indices()
        
        # 若当前类别不在占用量最大的类别中，则随机选择一个占用量最大的类别进行移除
        if cls not in largest_indices:
            largest = random.choice(largest_indices)
            # 获取目标类别中待移除实例的索引（默认随机选择，可通过重写get_target_index自定义策略）
            tgt_idx = self.get_target_index(self.data[largest][3])
            # 从目标类别的所有维度数据中移除对应索引的实例
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        # 若当前类别本身就是占用量最大的类别之一，则直接从当前类别中移除实例
        else:
            # 获取当前类别中待移除实例的索引
            tgt_idx = self.get_target_index(self.data[cls][3])
            # 从当前类别的所有维度数据中移除对应索引的实例
            for dim in self.data[cls]:
                dim.pop(tgt_idx)
        
        # 移除操作必定成功，返回True
        return True

    def reset_value(self, feats, cls, aux):
        self.data = [[[], [], [], []] for _ in range(self.num_class)]
        for i in range(len(feats)):
            tgt_idx = cls[i]
            self.data[tgt_idx][0].append(feats[i])
            self.data[tgt_idx][1].append(cls[i])
            self.data[tgt_idx][2].append(0)        # default domain=0
            self.data[tgt_idx][3].append(aux[i])   # confidence



def build_memory(args):
    """
    构造 memory 实例，根据 args.memory_type 调用不同策略。
    """
    memory_type = args.memory_type.lower()
    capacity = args.update_every_x  # 通常对应 memory buffer 的容量

    if memory_type == 'fifo':
        return FIFO(capacity)
    elif memory_type == 'reservoir':
        return Reservoir(capacity)
    elif memory_type == 'pbrs':
        return PBRS(capacity, num_class=get_num_classes(args))
    elif memory_type == 'hus':
        return HUS(capacity, num_class=get_num_classes(args), threshold=args.hus_threshold if hasattr(args, 'hus_threshold') else None)
    else:
        raise ValueError(f"Unsupported memory type: {args.memory_type}")

def get_num_classes(args):
    print(args.dataset, 'args.dataset')
    if args.dataset == "uci":
        num_classes = 6
    elif args.dataset == 'unimib':
        num_classes = 17
    elif args.dataset == 'oppo':
        num_classes = 17
    elif args.dataset == 'pamap2':
        num_classes = 12
    elif args.dataset == 'usc':
        num_classes = 12
    else:
        print('not this dataset')

    return num_classes