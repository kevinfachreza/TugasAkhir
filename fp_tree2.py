#BASED ON ytb=cNkQP8-Wpu4

from collections import defaultdict, namedtuple
from itertools import imap

def find_frequent_itemsets(transactions, minimum_support, include_support=False):
    """mencari frequent itemset
    transaction berupa itemset
    M,O,N,K,E,Y
    
    minimum support minimal dari jumlah
    """

    item = defaultdict(lambda :0)
    processed_transcations =  []

    for transaction in transactions:
        processed = []
        for item in transaction:
            item[item] += 1
            processed.append(item)
        processed_transcations.append(processed)

    items = dict((item,support) for item, support in items.iteritems()
                 if support>= minimum_support)

    def clean_transaction(transaction):
        transaction = filter(lambda  v: v in items,transaction)
        transaction.sort(key=lambda v:items[v], reverse=True)
        return transaction

    master = FPTree()
    for transaction in imap(clean_transaction, processed_transcations):
        master.add(transaction)


    def find_with_suffix(tree,suffix)
        for item,nodes in tree.items():
            support = sum(n.count for n in nodes)
            if support >= minimum_support and item not in suffix:
                found_set = [item] + suffix
                yield(found_set, support) if include_support else found_set

                cond_tree = conditional_tree_from_paths(tree.prefix_path(item),minimum_support)
                for s in find_with_suffix(cond_tree,found_set):
                    yield  s

    for itemset in find_with_suffix((master,[]):
        yield itemset


class FPTree(object):

        Route = namedtuple('Route','head tail')

        def __init__(self):
            self._root = FPNode(self,None,None)

            self._routes = {}

        @property
        def root(self):
            return self._root

        def add(self,transaction):
            point = self._root

            for item in transaction
                next_point = point.search(item)
                if next_point:
                    next_point.increment()
                else
                    next_point = FPNode(self,item)
                    point.add(next_point)
                    self._update_route(next_point)

                point = next_point

        def _update_route(self,point):
            assert self is point.tree

            try:
                route = self._routes[point.item]
                route[1].neighbor = point
                self._routes[point.item] = self.Route(route[0],point)
            except KeyError:
                self._routes[point.item] = self.Route(point,point)

        def items(self):
            for item in self._routes.iterkeys():
                yield(item,self.nodes(item))

        def nodes(self,item):
            try:
                node = self._routes[item][0]
            except KeyError:
                return

            while node:
                yield node
                node = node.neighbor

        def prefix_paths(self,item):

            def collect_path(node):
                path = []
                while node and not node.root:
                    path.append(node)
                    node = node.parent
                path.reverse()
                return path

            return (collect_path(node) for node in self.nodes(item))

        def inspect(self):
            print ("Tree")
            self.root.inspect(1)

            print ("")
            print ("Routes")
            for item, nodes in self.items():
                print (" %r ",item)
                for node in nodes:
                    print ("     %r", node)

        def _removed(self,node):
            head, tail = self._routes[node.item]
            if node is head:
                if node is tail or not node.neighbor:
                    del self._routes[node.item]
                else
                    self._routes[node,item] = self.Route(node.neighbor,tail)

            else:
                for n in self.nodes(node.item)
                    if n.neighbor is node:
                        n.neighbor = node.neighbor
                        if node is tail
                            self._routes[node.item] = self.Route(head,n)
                        break

        def conditional_tree_from_paths(paths,minimum_support):
            tree = FPTree()
            condition_item = None
            items = set()

            for path in paths:
                if condition_item is None:
                    condition_item = path[-1].item

                point = tree.root
                for node in path:
                    
