class ItemTimeStamp(object):
    def __init__(self, item):
        self.item = item

    def get_timestamp(self, item):
        return self.item.timestamp

class TimeStampQueue(object):
    def __init__(self):
        self.num_items = 0
        self.items_timestamps = {}

    def increment(self):
        self.num_items += 1

    def get_num_items(self):
        return self.num_items

    def add_item(self, itemtimestamp):
        self.items_timestamps.update()

# class Deck(object):
#     def __init__(self):
#         # Dictionary of timestamps => keep track of the state of a deck at time
#         self.deck = {}
#
#     def push(self, timestamp):
#         if timestamp not in self.deck:
#             self.deck.update({timestamp: TimeStampQueue()})
#             self.deck[timestamp].
#             self.deck[timestamp].increment()
#
#         else:
#
#     def pull(self):
#         pass
