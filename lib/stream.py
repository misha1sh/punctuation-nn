from collections import deque
import random

#https://stackoverflow.com/a/15993515
class ListRandom(object):
    def __init__(self):
        self.items = []

    def add_item(self, item):
        self.items.append(item)

    def remove_item(self, position):
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item

    def __len__(self):
        return len(self.items)

    def pop_random(self):
        assert len(self.items) > 0
        i = random.randrange(0, len(self.items))
        element = self.items[i]
        self.remove_item(i)
        return element

class Stream:
    def __init__(self, generator):
        try:
            self.generator = iter(generator)
        except TypeError:
            self.generator = generator

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)

    @staticmethod
    def repeat(element, n):
        def generator():
          for i in range(n):
              yield element
        return Stream(generator())

    def buffered_mix(self, elements_in_buffer_count):
        def generator():
            buffer = ListRandom()
            it = iter(self)
            while True:
                while len(buffer) < elements_in_buffer_count:
                    try:
                        buffer.add_item(next(it))
                    except StopIteration:
                        while len(buffer) > 0:
                            yield buffer.pop_random()
                        return
                yield buffer.pop_random()
        return Stream(generator())


    @staticmethod
    def mix_streams(streams, weights):
        def generator():
            iters = [iter(i) for i in streams]
            choices = list(range(len(streams)))
            i = 0
            while True:
                try:
                    i = random.choices(choices, weights)[0]
                    yield next(iters[i])
                except StopIteration:
                    weights[i] = 0
                    if sum(weights) == 0:
                        return
        return Stream(generator())


    def chain(self, another_stream):
        def generator():
            for i in self:
                yield i
            for i in another_stream:
                yield i
        return Stream(generator())

    def slide_window(self, window_size):
        res = deque()
        for i in self:
          res.append(i)
          if len(res) == window_size:
            yield Stream(res)
            res.popleft()

    def skip(self, count):
        def generator():
            n = count
            for i in self.generator:
                n -= 1
                if n == 0: break
            for i in self.generator:
                yield i
        return Stream(generator())

    def get(self, count):
        res = []
        for i in self:
            res.append(i)
            if len(res) == count:
                return res
        return res

    def limit(self, count):
        def generator():
            n = count
            for i in self.generator:
                yield i
                n -= 1
                if n == 0: break
        return Stream(generator())

    def map(self, func):
        def generator():
            for i in self.generator:
                yield func(i)
        return Stream(generator())

    def starmap(self, func):
        def generator():
            for i in self.generator:
                for j in func(i):
                    yield j
        return Stream(generator())

    def group(self, n):
        def generator():
            grouped = []
            for i in self.generator:
                grouped.append(i)
                if len(grouped) >= n:
                    yield grouped
                    grouped = []
            if len(grouped) != 0:
                yield grouped

        return Stream(generator())