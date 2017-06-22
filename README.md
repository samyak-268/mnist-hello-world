# mnist-hello-world
A simple MLP classifier for the MNIST dataset. 
This was written to be presented as a "Hello World" tutorial for a Deep Learning study group.

The Torch7 MNIST data loader is a pre-requisite. To install, run ``luarocks install mnist``

To start training, run ``th main.lua -gpu 1 -numEpochs 10``

Feel free to change the value of the parameters. Set the ``gpu`` param to 1,
only if ``cutorch`` and ``cunn`` are installed. Otherwise, use the default value of 0.
