import nn
from sklearn.model_selection import train_test_split

def read_wdbc(filename):
   xs = []
   ys = []
   try:
      with open(filename, 'r') as file:
            content = file.read().strip() # Read all content and remove leading/trailing whitespace
            if not content:
               print(f"File '{filename}' is empty or contains only whitespace.")
               return

            # Split the content by the specified separator
            # Filter out any empty strings that might result from splitting (e.g., extra newlines)
            rows = [s for s in content.split('\n') if s.strip()]

            for r in rows: 
               x = [s for s in r.split(',') if s.strip()]
               x.pop(0)
               y = x.pop(0)
               xs.append( [float(i) for i in x] )
               ys.append( 1 if y == 'M' else -1 )
   except FileNotFoundError:
      print(f"Error: File '{filename}' not found.")
   except Exception as e:
      print(f"An unexpected error occurred: {e}")
   
   return [xs, ys]

xs, ys = read_wdbc('wdbc/wdbc.data')

X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.25, random_state=34)

def wdbc_train(xs, ys):
   # n = nn.MLP(30, [8,8,1]) 
   n = nn.MLP(filename='wdbc/p2.txt') 
   n.calc_print(xs)
   n.train(xs, ys, 10, method=1, train_step=0.001, log=True) # method=1 daje bardzo dobre wyniki na poziomie 0.9999 - ale czy to przetrenowanie?
   n.calc_print(xs)
   n.save_to_file('wdbc/p2.txt') # zapisanie parametrow w pliku


def wdbc_test(xs, ys):
   n = nn.MLP(filename='wdbc/p2.txt') 
   # n = nn.MLP(30, [6,6,1]) 
   ypred = [1 if y.data >= 0 else -1 for y in n.calc(xs)]
   acc = 0
   count = len(ys)
   for yp, y in zip(ypred, ys):
      if yp == y : acc += 1
   
   print(f"accuracy = {acc/count}")

# wdbc_train(X_train, y_train)
wdbc_test(X_test, y_test)
# na powstałej sieci neuronowej udało sie wytrenowac model do skutecznosci 0.86

