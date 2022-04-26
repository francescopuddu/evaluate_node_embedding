# Test Node Embeddings 

Following the standard testing approach from the field, this component implements a classification algorithm to evaluate the output of a node-embedding model when labeled data is available. Results can be visualized either on the command line or with Tensorboard. 

## High level description
- Inputs:  the feature map (node --> latent vector), the labels map (label --> nodes) and the desired classification metric
- Output: the value of the selected metric at different percentage of nodes used for training 

Inspired by known implementations [1, 2], the module use the embeddings and labels of a random subset of the nodes to train a one-vs-all logistic regression classifier, that is then used to test the embeddings of the remaining nodes. 
The classification metric can be any variant of the F1 score. It can be selected in the configuration file using the [naming convention](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) of scikit-learn. 
<br/>
<br/>

## Input format
The input files are essentially the abovementioned two maps and the configuration file. Please note that examples are provided. 

#### Feature Map
Following the Word2Vec format, the header specifies the number of nodes and the number of features; each row starts with the ID of the node and contains the coordinates of the latent vector. 
Please note that all numbers are separated by a single space. 
<pre>
3 4
0 1.5 2.5 3.5 4.5
1 0.32 0.5 2.5 3 0.8
2 9.0 0.45 0.3 1.7 3.5
</pre>
<br/>

#### Labels Map
Each line starts with the identifier of the label and contains the list of the nodes belonging to that category. 
Please note that all numbers are separated by a single tab. 
<pre>
0    87    43    2    41
1    12    76
2    0    88    16
</pre>
<br/>

#### Configuration File
The json file storing the parameters for the classifier is expected to contain the following fields: 
- The path to the text file containing the embeddings in the specified format
- The path to the text file containing the labels in the specified format
- The number of shuffles to perform on data during the evaluation of the classification metric
- The selected classification metric (for the options refer to the scikit-learn documentation linked above)
- The name for the experiment in case the visualization on Tensorboard is preferred (leaving this field empty is equivalent to choosing to visualize the output on the command line)
<pre>
{
    "embedding_path": "path/to/embedding/file",
    "labels_path": "path/to/labels/file",
    "shuffles": 4,
    "metric" : "micro",
    "experiment_name" : "abc"
}
</pre>
<br/>

### Usage guidelines
To set up the environment, a requirements.txt file is provided. 
The module can be used by simply running "classifier.py". The only requested command line argument is the path to the configuration file, if none is provided then the script will look for "config.json" in the current directory by default. 


The repository provides the input files to run the module on example data. The embeddings come from the reference implementation of ProNE [3] on the well known Cora benchmark [4]. 

To run the example: 
<pre>
python classifier.py
</pre>

The correct output should look similar to this: 

On Tensorboard: 
![tensorboard](https://i.ibb.co/jMHhL5G/Immagine-2022-04-26-142236.png "Tensorboard")

On command line: 
<pre>
-------------------
Train percent : metric value
0.1 : 0.613515176374077
0.2 : 0.7305029995385326
0.3 : 0.7568565400843882
0.4 : 0.7710769230769231
0.5 : 0.7860044313146233
0.6 : 0.7972785977859779
0.7 : 0.8016605166051661
0.8 : 0.8058118081180811
0.9 : 0.7952029520295203
</pre>

<br/>
<br/>

### References
[1] https://arxiv.org/abs/1803.04742

[2] https://arxiv.org/abs/1906.06826

[3] https://github.com/THUDM/ProNE

[4] https://paperswithcode.com/dataset/cora
