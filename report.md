---
title: __Semantic Parsing of Nested Intents Using a Deep Neural Top-Down Algorithm__
author: |
	```{=latex}
	Arthur Amalvy \\ \texttt{108522605} \and Tran Cong Nghi \\ \texttt{108582603}
	```
header-includes:
- |
	```{=latex}
	\usepackage{float}
	\let\origfigure\figure
	\let\endorigfigure\endfigure
	\renewenvironment{figure}[1][2] {
		\expandafter\origfigure\expandafter[H]
	} {
		\endorigfigure
	}
	\makeatletter
	\let\oldlt\longtable
	\let\endoldlt\endlongtable
	\def\longtable{\@ifnextchar[\longtable@i \longtable@ii}
	\def\longtable@i[#1]{\begin{figure}[t]
		\onecolumn
		\begin{minipage}{0.5\textwidth}
		\oldlt[#1]
	}
	\def\longtable@ii{\begin{figure}[t]
		\onecolumn
		\begin{minipage}{0.5\textwidth}
		\oldlt
	}
	\def\endlongtable{\endoldlt
	\end{minipage}
	\twocolumn
	\end{figure}}
	\makeatother
	```
abstract: As intelligent assistants become more commons, they pose new challenges when it comes to interacting with users. To be able to carry an instruction, such systems usually rely on a mechanism of intent classification and slot detection. However, such a mechanism doesn't allow working with nested queries. To allow for representing such queries, one might use a tree like structure (such as _Gupta et al., 2018_ __[2]__). In this paper, we propose a new model able to parse queries into such a structure. Notable contributions include a novel recursive algorithm inspired by constituency parsing (_Stern et al., 2017_ __[1]__), and the use of SpanBERT (_Joshi et al., 2019_ __[4]__) to reprensent spans of text. Our model achieves results close to the state of the art in certain aspects, but falls short of beating it. However, considering some of the issues of the model are known and fixable, we still consider those results as promising. 
---

# Introduction

With the rise of intelligent assistants, understanding and correctly carrying an user query becomes an important topic. Semantic parsing of user queries is usually done by firstly classifying the intention of the user (which we call its _intent_), and secondly by identifying spans of texts corresponding to the parameters of the query (we call those parameters _slots_, and their actual occurence in the text _values_). However, this method is ineffective when parsing an input containing multiple queries, where the main query may be dependant on other queries. To solve this problem, _Gupta et al., 2018_ __[2]__ introduced a new representation of queries as trees, where slots can be considered as nested queries. Their model uses a Recurrent Neural Network Grammar (RNNG) (_Dyer et al., 2016_ __[5]__) to construct a tree by reading the input query token by tokens, and was further improved in _Einolghozati et al., 2019_ __[7]__. In this paper, we wanted to see if others kinds of algorithms would give better results. In order to do that, we designed and implemented a model inspired by the top-down parsing algorithm of _Stern et al., 2017_ __[1]__, originally meant for constituency parsing.\footnote{We release our code under the MIT license at https://github.com/Aethor/nintent}


# Methodology

We propose a novel algorithm for performing nested semantic parsing of an user query. Our goal is, given an user order, to produce a tree capturing all nested queries in this input. The characteristic of this tree are exactly the same as in _Gupta et al., 2018_ __[2]__, except in our implementation we do not allow text nodes : each node contains all the text of its span (see _figure 1_ as an example).  As outlined in in _Gupta et al., 2018_ __[2]__, such a tree must respect the following rules (which we modified to take into account our modified implementation) :

> * _The top level node must be an intent_
> * _An intent can only have slots as children_
> * _A slot can only have one intent as a child_

Our algorithm construct trees recursively from a top-down manner. Respecting such rules means that an alternation pattern has to be implemented : the top node is an intent, its children are slots, a potential slot child is an intent, this intent children are slots, _etc_... Therefore, we can divide our algorithm in two distinct stages.

![Representation of a query under our model](./assets/representation.png)


## Intent Level

![Our parsing model](./assets/model.png)

When the input of the algorithm is an intent (for example, at the top level), our algorithm performs two tasks :

* Classifying the intent
* Detecting slots

Intent classification is performed using a simple module composed of a fully connected neural network layer followed by a softmax activation function. This module takes as input the span representation (see the _span representation_ section) of the intent's text, and produces a probability distribution over all possible intent types.

Slot detection is the trickiest part of the algorithm. for an intent's text consisting of $t$ tokens, each n-gram (with $n = t - 1$) is considered as a candidate slot. Our model therefore learns to give a score to each n-gram (we further denote the score of the $i^{th}$ n-gram as $s(N_i)$). To do so, a different neural network is trained for each intent type\footnote{as we noticed during our experimentations, different intent types usually have different slots, therefore training a different model per intent type greatly enhances results}, and is used in conjonction to a softmax function to rate each n-gram by producing a probability distribution over two classes (the negative class meaning the span is not considered a slot, the positive one meaning the span should be considered a slot). The problem of such a method is that conflicts may arise when two overlapping n-grams are chosen, which means one must choose a way to resolve those conflict to choose only non-overlapping spans as slots. To do so, our implementation iterates over all n-grams, adding the $i^{th}$ n-gram in a list of detected slots if the following conditions are satisifed : 

* $s(N_i) \geq 0.5$ (in our implementation, as we use a softmax as the activation function, the score of the positive class is between 0 and 1)
* $s(N_i) \geq \frac{|O|}{\sum_j^{|O|}{\frac{1}{s(O_j)}}}$, the right function representing the harmonic mean of all overlapping spans $O$ of the $i^{th}$ n-gram\footnote{We instinctively use the harmonic mean to prevent spans with low scores to be chosen, although other methods are possible. We experimented with several other functions, but found only minor differences in results}.

At this level, the algorithm returns a tree whose root is the current intent, and having as children all results of the algorithm recursively launched on detected slots.

## Slot Level

When the input of the algorithm is a slot, our algorithm also performs two tasks :

* Classifying the slot
* Checking wether or not the current slot is a nested intent

To classify the current slot, we also use a simple module composed of a fully connected neural network layer followed by a softmax function. However, for the same reason as slot detection at the intent level, we train a different network for each possible parent intent type, greatly enhancing classification performance.

Checking if a slot is also a nested intent is, once again, performed by a fully connected layer followed by a softmax function.

At this level, the algorithm returns a tree whose root is the current slot, and having as child an intent if it was found to be nested, or being a leaf node otherwise.


## Span Representation

Every time a module consisting of a fully connected neural network layer has been mentioned above, its input is a span representation. Such a span representation is obtained using a pre-trained SpanBERT model __[4]__, that is finetuned during training. SpanBERT is a modified version of BERT, where training is done by making the model predict a span content using only its boundary tokens, and is therefore very well suited to reprensent spans of text. 


## Training

The training algorithm is slightly different of the inference one.  At training time, we compare the generated tree with the gold tree to generate a loss. To train our slot-detection neural networks more efficiently, we do not solve conflicts between overlapping spans : instead, for each span score, we compare it with the score of this span in the gold tree (either 0 or 1)

At the intent level, the loss is a sum of :

* for the intent classification task, a cross entropy loss between the outputed probability distribution over possible intent types and the gold tree true intent type
* for the slot detection task, the sum of all cross entropy loss between span scores and the gold tree scores (0 or 1). The negative class of this cross entropy loss has weight $w_d = \frac{N}{N_g}$, where $N$ is the number of t-1-gram in the sentence with $t$ tokens, and $N_g$ the number of slots in the gold tree. 

At te slot level, the loss is a sum of :

* for the slot classification task, another cross entropy loss between the outputed probability distribution over all possible slot types and the gold tree true slot type 
* for the intent detection task, a cross entropy between the outputed score and the true score (0 if the gold tree doesn't contain a nested intent, 1 otherwise)


When training, we do not follow the current model slot or intent choice when constructing the tree : instead, we rely on the information of the gold tree, effectively performing teacher forcing. 


# Experiments

## Dataset

To compare our work with _Gupta et al., 2018_, we use their proposed dataset. It is composed of 44783 user queries, with 25 intent types and 36 slot types. The dataset focus is on navigation, events and navigation to events. Trees in this dataset have a mean depth of 2.54, while user queries have a mean length of 8.93 tokens. 35% of trees have a depth of more than 2 (which indicates a nested query). The dataset is split into training, validation and testing set with 70:10:20 ratio.

![Dataset statistics](./assets/dataset.png)


## Evaluation Metrics

Again, to compare our work with _Gupta et al. 2018_, we mainly use two metrics :

* Exact accuracy : Exact accuracy is the ratio of predicted trees that are _exactly equals_ to gold trees. This is a very harsh metric.
* Labeled Bracketing Precision, Recall and F1 : This metric is based of the number of spans correctly labeled. Labeled Bracketing Recall is based on the number of spans of the gold tree present and correctly labeled in the predicted tree, while Labeled Bracketing Precision is based on the number of correcty predicted spans labels of the predicted tree.


## Results

In the following table, the _Gupta et al., 2018_ __[2]__ results are labeled _original_, while the _Einolghozati et al., 2019_ __[7]__ results are labeled _enhanced_\footnote{for both of these papers, we kept only the best result}.

| Model      | Exact Accuracy |
|:----------:|:--------------:|
| _original_ |     78.51      |
| _enhanced_ |     87.25      |
| _ours_     |     45.13      |

|    Model   | LB-Precision | LB-Recall |  LB-F1    |
|:----------:|:------------:|:---------:|:---------:|
| _original_ |     90.62    |    89.84  |  90.23    |
| _enhanced_ |  _unknown_   | _unknown_ | _unknown_ |
| _ours_     |     95.36    |    84.69  |  89.71    |

Our results are pretty close to those of _Gupta et al., 2018_. when talking about the bracketing labeled metrics. We attributes our high bracketing labeled precision to the use of SpanBERT to classify spans of text.

Sadly, our model lags behind in terms of exact accuracy and recall. To understand why, we performed a quick error analysis, and found out the type of situations mishandled by our model. When a nested intent is detected, and the full span of that nested intent should be considered a slot, our model does not pick up that slot (_figure 1_ is a perfect example of this situation). This may be due to our model having no information about the structure of the tree, but we didn't have time to push this hypothesis further. Another possible issue may come from the fact that the algorithm is never confronted to his mistakes at training time, because we always follow informations from the gold tree, which may cause poor performances at test time. This may have been solved by using a dynamic oracle strategy at training time (as in _Cross and Huang, 2016_ __[6]__) to allow for exploration. 


# Conclusion

In this work, we proposed a novel method to parse user orders into a nested query tree, inspired by work on constituency parsing. Our results are close to the state of the art when measured with bracketing labeled metrics, although we lag behind in terms of exact accuracy. Despite this, we believe that by fixing the apparent problems of the model, exact accuracy and bracketing labeled recall of our model should improve, and get close to _Gupta et al., 2017_ work or even beat it. 

\newpage
\onecolumn

# References

* __[1]__ _Stern M., Andreas J., Klein D. A Minimal Span-based neural constituency parser. arXiv preprint arXiv:1705.03919. 2017._
* __[2]__ _Gupta S., Shah R., Mohit M., Kumar A., Lewis M. Semantic Parsing for Task Oriented Dialog Using Hierarchical Representations. arXiv preprint arXiv:1810.07942. 2018._
* __[3]__ _Einolghozati A., Pasupat P., Gupta S., Shah R., Mohit M., Lewis M., Zettlemoyer L. Improving Semantic Parsing for Task Oriented Dialog. arXiv preprint arXiv:1902:06000. 2019._
* __[4]__ _Joshi M., Chen D., Liu Y., Weld D. S., Zettlemoyer L., Levy O. Spanbert : Improving Pre-training by Representing and Predicting Spans. arXiv preprint arXiv:1907.10529. 2019._
* __[5]__ _Dyer C., Kuncoro A., Ballesteros M., Smith N. A. Recurrent Neural Network Grammars. arXiv preprint arXiv:1602:07776. 2016._
* __[6]__ _Cross J., Huang L. Span-Based Constituency Parsing with a Structure-System and Provably Optimal Dynamic Oracle. arXiv preprint arXiv:1612:06475. 2016_
* __[7]__ _Einolghozati A., Pasupat P., Gupta S., Shah R., Mohit M., Lewis M., Zettlemoyer L. Improving Semantic Parsing for Task Oriented Dialog. arXiv preprint arXiv:1902.060000. 2019._
