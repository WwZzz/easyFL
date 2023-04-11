
# How to use the Dcokerfile


- Step 1

Modify the Dockerfile accorrding to the comment to build a basic pytorch environment 

- Step 2

Build the images using the command: 
> $ docker build -t flgo .

- Note

- Step 3

Create a container using the command:

> $ docker run -itd --gpus all --network=host flgo /bin/bash

