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
        # Make sure the input instance contains 4 elements: feature, pseudo-label, domain information, confidence
        assert len(instance) == 4
        # Extract the pseudo-label of the instance, used for subsequent categorized storage
        cls = instance[1]
        # Increment the count of instances encountered for the current pseudo-label class
        self.counter[cls] += 1
        # Initialize the add flag, defaulting to True (addition allowed)
        is_add = True

        # If a confidence threshold is set and the confidence of the current instance is below the threshold, reject the addition
        if self.threshold is not None and instance[3] < self.threshold:
            is_add = False
        # If the memory capacity is already full, try to remove an instance to make room; the removal result decides whether a new instance may be added
        elif self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        # When addition is allowed, store each component of the instance into the memory list of the corresponding class
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
        Remove an instance from memory, preferring to remove from the class with the largest occupancy, in order to keep the memory capacity balanced
        
        Args:
            cls: pseudo-label class of the instance currently to be added
        Returns:
            bool: whether the removal operation succeeded (always True here, since the removal logic guarantees that at least one instance is removed)
        """
        # Get the list of indices of the classes with the largest current occupancy (there may be several classes with the same maximum occupancy)
        largest_indices = self.get_largest_indices()
        
        # If the current class is not among the classes with the largest occupancy, randomly choose one of those classes to remove from
        if cls not in largest_indices:
            largest = random.choice(largest_indices)
            # Get the index of the instance to remove in the target class (randomly chosen by default; the strategy can be customized by overriding get_target_index)
            tgt_idx = self.get_target_index(self.data[largest][3])
            # Remove the instance at the corresponding index from all dimension data of the target class
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        # If the current class is itself one of the classes with the largest occupancy, remove the instance directly from the current class
        else:
            # Get the index of the instance to remove in the current class
            tgt_idx = self.get_target_index(self.data[cls][3])
            # Remove the instance at the corresponding index from all dimension data of the current class
            for dim in self.data[cls]:
                dim.pop(tgt_idx)
        
        # The removal operation always succeeds, return True
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
    Construct a memory instance, calling a different strategy according to args.memory_type.
    """
    memory_type = args.memory_type.lower()
    capacity = args.update_every_x  # usually corresponds to the capacity of the memory buffer

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