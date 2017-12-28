This is a small package to download and parse out the [MNIST Digits Dataset](http://yann.lecun.com/exdb/mnist/) from Yann LeCun's website. Although it's straightforward to download the files in a browser, this handles the bit parsing so the data can easily be used in python with minimal overhead.

**Usage**

    import mnist

    m = mnist.Mnist(dataset='train')

    m.download(destdir='/tmp')

    y, x = m.read(output='pandas')
    
    # y, x = m.read(output='numpy')
