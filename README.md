This is a small package to download and parse out the [MNIST Digits Dataset](http://yann.lecun.com/exdb/mnist/) from Yann LeCun's website. Although it's straightforward to download the files in a browser, this handles the bit parsing so the data can easily be used in python with minimal overhead.

**Usage**

    import mnist

    m = mnist.Mnist(dir='/tmp/mnist')

    x_train = m.xtrain
    y_train = m.ytrain
    x_test  = m.xtest
    y_test  = m.ytest
    
    # To load batches from a generator
   
    # dataset one of 'train', 'test', all 
    batch = m.batch(batch_size=1000, dataset='train')
    
    y, x = next(batch)
    
