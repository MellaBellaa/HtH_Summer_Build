# Instructions  
In this assignment, we're going to set up a basic machine learning algorihtm that can learn the difference between two categories of sentences, positive and negative. For example, if you had the sentence "I love it!", we want to train a machine to know that this sentence is associated with happy and positive emotions. If we have a sentence like "it was really terrible", we want the machine to label it as a negative or sad sentence.

We will make this process much simpler by using the Python library scikit-learn. It makes it possible to do advanced machine learning in just a few lines of code. At the end of this tutorial, you'll understand the fundamental ideas of some basic machine learning processes and have a program that can learn by itself to distinguish between different categories of text. 

In this assignment, we will:

- Create some simple mock data - text to classify as positive or negative
- Explain vectorisation of the dataset
- Cover how to classify text using a machine learning classifier
- Compare this to a manual classifier
- NOTE: All the things you need to do will be outlined and explained here in the instructions, so make sure you are reading this thoroughly.

## Steps
1| open the "main.py" file and add the following two imports to the top. Run the Repl to be sure your dependencies are installed. Line 1 imports the tree module, which will give us a Decision Tree classifier that can learn from data. Line 2 imports a vectoriser -- something that can turn text into numbers. Line 3 gives us access to something we are already familiar with, matplotlib.
:-:|:-
```python
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
``` 
2| Before we get started, we'll need to create a very simple dataset to act as our mock data. Add the following lines of code to your "main.py" file.
:-:|:-
```python
positive_texts = [
    "we love you",
    "they love us",
    "you are good",
    "he is good",
    "they love mary"
]

negative_texts =  [
    "we hate you", 
    "they hate us",
    "you are bad",
    "he is bad",
    "we hate mary"
]

test_texts = [
    "they love mary",
    "they are good",
    "why do you hate mary",
    "they are almost always good",
    "we are very bad"
]

'''
Here are three simple datasets. The first one contains five positive sentences; the second one contains five negative sentences; and the last contains a mix of both.

We can easily see which sentences are positive and which are negative, but can we teach a computer to do this?

We'll use the two lists of positive and negative sentences to train a model with Python. We will give examples to the computer that are already labelled as positive or negative. The computer will compute how to find the difference, and then we'll test it with our mixed sentences. The computer will guess whether each example is positive or negative.
'''
``` 
VECTORIZATION| The first step in many machine learning algorithms is to translate your data into a format that makes sense to a computer. In the case of language and text data,this is done through a process called vectorization.This is when each unique word in the dataset is given a number, from 0 onwards. Each text can then be represented by an array of numbers, representing how often each possible word appears in the text. This process is known as vectorization.
:-:|:-

```python
# Vectorization Example: Let's say we have two sample lists:

sample_sentences = [["nice pizza is nice"], ["what is pizza"]]

# First we go through can find all the unique words and give them a number. Those words would be "nice", "pizza", "is", and "what"

unique_words = {
    "nice": 0,
    "pizza": 1,
    "is": 2,
    "what": 3
}

# Next we go through our sentences from left to right and create vectors based on how many times each of the unique words in our vocabulary appears in that particular sentence

vectors = [[2, 1, 1, 0], [0, 1, 1, 1]]

# this is showing us that in our first sentence, the word nice appears twice, pizza appears once, is appears once and what does not appear at all. In our second sentence, nice does not appear, while all the other three appear just once
```

The "Bag of Words"| Each sentence vector is the same length as the total of unique words in the dataset. This means that your vector length can be longer than the actual sentence. We can see this with our second sentence; It has three actual words but its vector has four entries. With real data, these list get very long. There are more unique words than one would want to count in most datasets, so each sentence vector would be very long. Each position in the list represents a unique word, and each value represents how often that word appears in that sentence. This representation is called "bag of words" because we lose all of the information represented by the order of words. We don't know where the word nice is in our first sentence but we know its there twice.
:-:|:-

3| Now that we know a little more about the process of vectorization, let's create our own bag of words with our positive and negative sentences. Add the following lines of code to your "main.py"
:-:|:-

``` Python
training_texts = negative_texts + positive_texts

training_labels = ["negative"] * len(negative_texts) + ["positive"] * len(positive_texts)

'''
Here we are creating the labels that catagorize each sentence as either positive our negative. Here is what our two new variables look like:

training_text = ['we hate you', 'they hate us', 'you are bad', 'he is bad', 'we hate mary', 'we love you', 'they love us', 'you are good', 'he is good', 'they love mary']

training_labels = ['negative', 'negative', 'negative', 'negative', 'negative', 'positive', 'positive', 'positive', 'positive', 'positive']

if we wanted to grab each sentence and its appropriate label we could use the same index position in each variable:

first_sentence = (training_text[0] training_labels[0])
'''
```

4| Now let's use the vectorizer we imported earlier from Python's scikit-learn. Add the following lines of code to your "main.py":
:-:|:-

``` Python
vectorizer = CountVectorizer()

vectorizer.fit(training_texts)

unique_words = vectorizer.vocabulary_

'''
Here we can already see how Python makes this process much easier. Instead of having to go through and hand pick all our unique words, the scikit-learn vectorizer handles that for us. You can use a print() statement to see your unique words. It should look something like this:

uniqe_words = {
    'we': 10,
    'hate': 3, 
    'you': 11, 
    'they': 8, 
    'us': 9, 
    'are': 0, 
    'bad': 1, 
    'he': 4, 
    'is': 5, 
    'mary': 7, 
    'love':6, 
    'good': 2}

The only difference between this and our earlier example is that the numbers we given based on the words alphabetically order.
'''
```

5| Now that we have a vectorizer and all our unique words, its time to transform our sentences into vectors. Add the following lines of code:
:-:|:-

``` Python
training_vectors = vectorizer.transform(training_texts)

testing_vectors = vectorizer.transform(test_texts)
```

CLASSIFICATION| A classifier is a machine learning model that predicts a label for a given input. In our case, the input is the sentence and the output is either "positive" or "negative". A classifier is trained by giving it labelled data and it tries to learn rules based on that data. Every time it gets more data, it updates its rules slightly to account for the new information.
:-:|:-

6| Since we now know a little more about what a classifier is, let's try to build one for our sentences. There are many kinds of classifier algorithms, but one of the simplest is called a Decision Tree. This is what we will be using with the tree import from scikit-learn. Add the following lines of code: 
:-:|:-

``` Python
classifier = tree.DecisionTreeClassifier()

classifier.fit(training_vectors, training_labels)

predictions = classifier.predict(testing_vectors)

print(predictions)

'''
When creating our classifier, we finally make use of our training_labels variable, so that our algorithm knows what sort of outputs it should be giving. Once we print our predictions we should see something like:

predictions = ['positive' 'positive' 'negative' 'positive' 'negative']

We would then compare there predictions to our original test sentences:

test_texts = ["they love mary", "they are good", "why do you hate mary", "they are almost always good", "we are very bad"]

It looks like the computer has learned well! The words "bad" and "hate" appear only in the negative texts and the words "good" and "love", only in the positive ones. Other words like "they", "mary", "you" and "we" appear in both. A well trained model will have learned to ignore the words that appear in both, and focus on "good", "bad", "love" and "hate".
'''
```

6| Now let's take a closer look at our tree classifier. Add these following lines of code to create a visualization of our model using matplotlib:
:-:|:-

``` Python
fig = plt.figure(figsize=(5,5))

tree.plot_tree(classifier,feature_names = vectorizer.get_feature_names(), rounded = True, filled = True)

fig.savefig('tree.png')

'''
One feature of matplotlib we haven't talked much about is the ability to save any visualizations you create! You can download visualizations as a regular image or a gif, depending on which is more appropriate. The savefig() method makes this quick and simple. You should see a file called 'tree.png' show up near your "main.py" in the file explorer. If you open it, you can see your tree graph.
'''
```

``` Python
# Compare this manually built classifier to your new machine learning tree algorithm

def manual_classify(text):
    if "hate" in text:
        return "negative"
    if "bad" in text:
        return "negative"
    return "positive"

predictions = []

for text in test_texts:
    prediction = manual_classify(text)
    predictions.append(prediction)
  
print(predictions)
```

CONCLUSION| When the task at hand is this simple, it's often easier to write a couple of rules manually rather than using Machine Learning. Our dataset was trivially simple, but a real-world dataset might need thousands or millions of rules, and while we could write the if-statements "by hand", it's much easier and faster if we can teach machines to learn these by themselves. Also, a set of manual rules can only work for a single dataset. However, a perfected machine learning model can be used for many things by changing the input data! In this example, our model was perfect and correctly classify all five unseen sentences, but this is not usually the case for real-world data. Machine learning models are based on probability; the goal is to make them as accurate as possible, but you will rarely, if ever, get 100% accuracy.
:-:|:-
