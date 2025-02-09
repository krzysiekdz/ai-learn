
class Value:
    def __init__(self, data):
        self.data = data 
    def __repr__(self):
        return f'Value({self.data})'
    def __add__(self, other):
        return Value(self.data + other.data)
    def __sub__(self, other):
        return Value(self.data - other.data)
    def __mul__(self, other):
        return Value(self.data * other.data)
    


# drzewo operacji z uzyciem klasy Value 
a = Value(4)
b = Value(-3)
c = Value(5)
print(a*c)
    
    