import pickle
import numpy as np

class IndexableSet:
    def __init__(self, max_count: int):
        self.dictionary = {}
        self.counter = 0
        self.max_count = max_count

    def add(self, number: int):
        if number not in self.dictionary:
            if self.max_count is not None and self.counter >= self.max_count:
                raise Exception("Maximum count of unique elements exceeded.")
            self.dictionary[number] = self.counter
            self.counter += 1
        return self.dictionary[number]

    def serialize(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump((self.dictionary, self.counter), file)

    @classmethod
    def deserialize(cls, file_path, max_count: int):
        with open(file_path, 'rb') as file:
            dictionary, counter = pickle.load(file)
        instance = cls(max_count)
        instance.dictionary = dictionary
        instance.counter = counter
        return instance

    def index_array(self, ary: np.array) -> np.array:
        return np.vectorize(lambda x: self.add(x))(ary)

    def reverse_index_array(self, ary: np.array) -> np.array:
        reverse_dict = {v: k for k, v in self.dictionary.items()}
        return np.vectorize(lambda x: reverse_dict.get(x, None))(ary)


if __name__ == "__main__":
    # Example usage:
    indexable_set = IndexableSet(max_count=3)
    assert indexable_set.add(2) == 0
    assert indexable_set.add(2) == 0
    assert indexable_set.add(3) == 1
    assert indexable_set.add(4) == 2

    # Serialize the object
    indexable_set.serialize('indexable_set.pkl')

    # Deserialize the object
    deserialized_set = IndexableSet.deserialize('indexable_set.pkl', max_count=3)
    print(deserialized_set.add(3))  # Output: 1

    try:
        print(indexable_set.add(5))  # This will raise an exception
    except Exception as e:
        print(e)  # Output: Maximum count of unique elements exceeded.