# Johannes Siedersleben, QAware GmbH, Munich, Germany
# 18/05/2020

# Augmenting classes after creation


class MyClass(object):      # an empty class
    q = 300
    pass


def init(self, x, y):       # an external constructor
    self.x = x
    self.y = y


def f(self):                # an external instance method
    return self.x - self.y


def g(self):
    return self.z


MyClass.__init__ = init     # assigning the constructor
MyClass.z = 600             # assigning a class variable
MyClass.sub = f             # assigning an instance method
MyClass.get = g

# all of the following works

c = MyClass(3, 4)
print(MyClass.q)   # 300
print(MyClass.z)   # 600
print(c.z)         # 600
print(c.sub())     # -1
print(c.get())     # 600
