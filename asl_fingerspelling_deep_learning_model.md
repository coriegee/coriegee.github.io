---
layout: inner
title: ASL Fingerspelling Recognition
permalink: /asl_fingerspelling_deep_learning_model/
---
<html>
<head>
  <style>
    h1 {
      font-size: 48px;
    }
  </style>
</head>
<body>
	<br>
	<br>
	<br>
	<div align="center">
	  <h1>ASL Fingerspelling Recognition with Transformer Neural Networks</h1>
	</div>
	<br>
	<br>
</body>
</html>

[View the full project on GitHub](https://github.com/coriegee/ASLRecognitionModel)

## Project Overview
In my pursuit of refining my data science skills, I embarked on an exciting project to build an ASL (American Sign Language) fingerspelling recognition model. The objective was simple yet impactful: accurately interpret ASL gestures from video sequences and output the interpretation into text. This blog post takes you through the ins and outs of my journey, highlighting key steps, code snippets, and insights gained along the way.  
<br>

## Understanding the Data
The project kicked off with a thorough understanding of the problem at hand: recognizing and interpreting ASL fingerspelling. ASL, a complex and expressive language, employs hand movements and positions to convey letters and words. The dataset was supplied by Google for a Kaggle challenge, it provided 67208 sequences of video frames of ASL Fingerspelling phrases contained within 68 parquet files. Also provided was a `train.csv` file which indexed the parquet files, sequences and labelled the phrases within them.  
<br>

As an example, let us take a look at the top 5 rows of the the `train.csv` file.
{% highlight python %}
# Load the dataset
dataset_df = pd.read_csv('data/train.csv')
dataset_df.head()
{% endhighlight %}  
![ASL Dataset Header](/img/posts/ASLFingerspelling/dataset_head.png)  
<br>

At this point its worth noting that the phrases in the dataset are not typical english sentences. The dataset mostly contains randomly generated addresses, phone numbers, and urls derived from components of real addresses/phone numbers/urls.
Now, let us open the parquet file of the first row and fetch the data for this particular `sequence_id`.
{% highlight python %}
# Fetch data from parquet file
sample_sequence_df = pq.read_table(f"/data/train_landmarks/{str(file_id)}.parquet",
    filters=[[('sequence_id', '=', sequence_id)],]).to_pandas()
sample_sequence_df.head()
{% endhighlight %}  
![ASL Landmarks Dataset Header](/img/posts/ASLFingerspelling/landmarks_head.png)  
<br>

This particular phrase (3 creekhouse) contains 123 entries (or frames) and 1630 columns including 1628 landmark coordinates. You'll notice this dataset contains all the landmarks of the human body. The landmarks which we are interested in for reading ASL are the hand landmarks and the pose landmarks. This image illustrates the landmarks on a hand.  
<br>

![Hand Landmarks Diagram](/img/posts/ASLFingerspelling/hand_landmarks.png)  
<br>

We can use Mediapipe to visualise an animation of the hand landmarks of a frame in the dataset as an example, see how the landmarks represent the hand in making various ASL gestures.  
<br>

![Dataset landmarks example](/img/posts/ASLFingerspelling/dataset_landmarks_example.gif)  
<br>

## Data Exploration
Let's start the data exploration by investigating the distribution of the target class, in this case the phrases. Here we're going to take a look at the distribution of characters in the phrases to see if we can learn anything.  
<br>

![Class distribution characters](/img/posts/ASLFingerspelling/class_distribution_characters.png)  
<br>

As expected the most abundant characters are vowels and spaces, and the least are special characters. If the model is trained on imbalanced data, it may become biased towards the majority classes and perform poorly on the minority classes. 

To address class imbalance, we could consider the following strategies:
* Oversampling
* Undersampling
* Class Weighting
* Transfer Learning
* Ensemble Methods

If we have time to improve the model further, we may implement some of these strategies later.

Next, we take a look at our frames dataset to see if it contains any _bad_ data. Here we find that often the number of frames used to represent a phrase is less than the number of characters in that phrase. For example, 7 frames may be used to represent 10 or more characters. The lowest possible number of frames to represent 10 characters would be 10 frames. This data issue could be caused by missing video frames. Let's do some investigation to quantify this by calculating frame to phrase ratio and plotting the distribution.
![Frames to phrase length ratio distribution](/img/posts/ASLFingerspelling/ftp_distribution.png)  
<br>

We can see that the distribution is normal other than a spike in the bucket 0 to 1. We need to filter out these phrases as they will likely reduce the performance of the model.  
<br>

## Preprocessing the Data
The bedrock of any triumphant machine learning endeavor is data preprocessing. For convenience and efficiency, we will rearrange the data so that each parquet file contains the landmark data for hands and pose along with the phrase it represents. This way we don't have to switch between `train.csv` and its parquet file.

We will save the new data in the TFRecord format. TFRecord format is a simple format for storing a sequence of binary records. Storing and loading the data using TFRecord is much more efficient and faster when working with TensorFlow.

We then also create a mapping of characters to integer values, including extra characters  "<" and ">" to mark the start and end of each phrase, and "P" for padding. Since the input data for a deep learning model must be a single tensor, we need to resize and/or add padding to make the individual samples in the sequence have the same lengths. 

Another piece of preprocessing that we take care of is identifying the dominant hand. Fingerspelling is usually done with the dominant hand, for which we do not have a label for. Thus the next step is to determine the dominant hand logically and reshape the data accordingly.  

{% highlight python %} 
def pre_process(x):
    # Extract relevant coordinates
    rhand = tf.gather(x, RHAND_IDX, axis=1)
    lhand = tf.gather(x, LHAND_IDX, axis=1)
    # ... other coordinate extraction ...

    # Determine dominant hand based on NaN values
    # ... dominant hand preprocessing ...

    # Normalize and reshape coordinates
    # ... coordinate normalization ...

    # Concatenate hand and pose data
    x = tf.concat([hand, pose], axis=1)
    x = resize_pad(x)
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    x = tf.reshape(x, (FRAME_LEN, len(LHAND_IDX) + len(LPOSE_IDX)))
    return x
{% endhighlight %}  
<br>

Lastly before building the Transformer model, we create functions to read the TFRecord data, transpose and apply masks to the landmark coordinates. We then vectorize the phrase corresponding to the landmarks using the character mapping, before converting it to Tensors. We then split the data in train and test samples, using 80% of the data to train the model with.  
<br>

## Building the Transformer Model
At the core of this project lies the Transformer model, an architecture renowned for processing sequential data efficiently. The model we are going to design is similar to the one used in the "Automatic Speech Recognition with Transformer" tutorial for Keras. We will finetune only a small part of the model since we can treat the ASL Fingerspelling recognition problem similar to speech recognition. In both cases, we have to predict a phrase from a sequence of data.

A transformer model is a neural network architecture used for tasks like speech recognition. It processes audio input by using self-attention mechanisms to capture relationships between different parts of the input sequence. With an encoder-decoder structure, multi-head attention, and feedforward networks, it learns patterns effectively. Transformers excel due to their capacity to handle long-range dependencies and their parallel processing nature, making them a potent choice for our ASL Fingerspelling recognition use case. The diagram below illustrates the similarities between speech recognition and ASL Fingerspelling recognition.
![Transformer Model Diagram](/img/posts/ASLFingerspelling/transformer_diagram.png)  
<br>


Let's break down how we build the Transformer model.  


#### TokenEmbedding class:
This class defines a layer that embeds tokens into continuous vectors. It uses an embedding layer along with positional encoding to create embeddings for tokens. The positional encoding helps the model differentiate the positions of tokens in the input sequence. The call method combines the embeddings and positional encodings for each input token.

{% highlight python %}
class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.num_hid = num_hid
        self.emb = tf.keras.layers.Embedding(num_vocab, num_hid)
        self.pos_emb = self.positional_encoding(maxlen-1, num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        x = tf.math.multiply(x, tf.math.sqrt(tf.cast(self.num_hid, tf.float32)))
        return x + self.pos_emb[:maxlen, :]
    
    def positional_encoding(self, maxlen, num_hid):
        depth = num_hid/2
        positions = tf.range(maxlen, dtype = tf.float32)[..., tf.newaxis]
        depths = tf.range(depth, dtype = tf.float32)[np.newaxis, :]/depth
        angle_rates = tf.math.divide(1, tf.math.pow(tf.cast(10000, tf.float32), depths))
        angle_rads = tf.linalg.matmul(positions, angle_rates)
        pos_encoding = tf.concat(
          [tf.math.sin(angle_rads), tf.math.cos(angle_rads)],
          axis=-1) 
        return pos_encoding
{% endhighlight %}  
<br>

#### LandmarkEmbedding class:
Similar to the TokenEmbedding class, this class also embeds tokens, but it includes additional convolutional layers. These convolutional layers process the input tokens in a way that can capture local patterns within the sequence. The positional encoding is again combined with the convolutional outputs to form the final embeddings.  

{% highlight python %}
class LandmarkEmbedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            num_hid, 11, padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            num_hid, 11, padding="same", activation="relu"
        )
        self.conv3 = tf.keras.layers.Conv1D(
            num_hid, 11, padding="same", activation="relu"
        )
        self.pos_emb = self.positional_encoding(maxlen, num_hid)
        self.maxlen = maxlen
        self.num_hid = num_hid

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = tf.math.multiply(x, tf.math.sqrt(tf.cast(self.num_hid, tf.float32)))
        x = x + self.pos_emb
        
        return x
    
    def positional_encoding(self, maxlen, num_hid):
        depth = num_hid/2
        positions = tf.range(maxlen, dtype = tf.float32)[..., tf.newaxis]
        depths = tf.range(depth, dtype = tf.float32)[np.newaxis, :]/depth
        angle_rates = tf.math.divide(1, tf.math.pow(tf.cast(10000, tf.float32), depths))
        angle_rads = tf.linalg.matmul(positions, angle_rates)
        pos_encoding = tf.concat(
          [tf.math.sin(angle_rads), tf.math.cos(angle_rads)],
          axis=-1) 
        return pos_encoding
{% endhighlight %}  
<br>

#### TransformerEncoder class:
This class defines a single layer of the transformer encoder. It consists of multi-head self-attention, feed-forward neural networks, and layer normalization. The call method performs self-attention, applies feed-forward networks, and returns the output of the encoder layer.  

{% highlight python %}
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
{% endhighlight %}  
<br>

#### TransformerDecoder class:
This class defines a single layer of the transformer decoder. It includes self-attention, encoder-decoder attention, feed-forward networks, and layer normalization. The call method applies the different attention mechanisms and feed-forward networks, returning the final output of the decoder layer.  

{% highlight python %}
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = layers.Dropout(0.5)
        self.enc_dropout = layers.Dropout(0.1)
        self.ffn_dropout = layers.Dropout(0.1)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Masks the upper half of the dot product matrix in self attention.

        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [batch_size[..., tf.newaxis], tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, enc_out, target, training):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.self_att(target, target, attention_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att, training = training))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out, training = training) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out, training = training))
        return ffn_out_norm
{% endhighlight %}   
<br>

We then complete the Transformer model, using the mean "edit_dist" (Levenshtein distance) to judge the performance of the model.  

{% highlight python %}
class Transformer(keras.Model):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        source_maxlen=100,
        target_maxlen=100,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=60,
    ):
        super().__init__()
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.acc_metric = keras.metrics.Mean(name="edit_dist")
		# ... complete transformer model ...
{% endhighlight %}   
<br>


## Training and Evaluation
For the models evaluation metrics, we will use the **normalised total Levenshtein distance** to measure the overall accuracy of the models predictions against the true sequences. 

The Levenshtein distance between two strings `a, b` (of length `|a|` and `|b|` respectively) is given by `lev(a, b)` where:
![Levenshtein Distance](/img/posts/ASLFingerspelling/levenshtein_distance.png)
[Source](https://en.wikipedia.org/wiki/Levenshtein_distance)  

For our model, let the total number of characters in all of the phrases be `N` and the total levenshtein distance be `D`. The **normalised total Levenshtein distance** is then:  

`(N - D) / N`
<br>

To maximise the normalised total Levenshtein distance, we will track the weighted mean Levenshtein distance between sequences of the model during training.  
<br>

Now that we can track our performance evalution measure, we can train the model and calculate the training loss and validation loss.

Optimizing training and validation loss is crucial in machine learning to ensure that our model generalizes well to unseen data. By minimizing training loss, the model learns to fit the training data accurately. Validation loss, on the other hand, helps us gauge the model's performance on new, unseen data. Balancing these losses prevents overfitting (when the model memorises the training data) and underfitting (when the model fails to capture patterns). In this case, we found our optimal training epochs to be 9.  
![Training loss and validation loss](/img/posts/ASLFingerspelling/training_validation_loss.png)  
<br>

## Visualizing Predictions
Let's take a look at some of the models predictions of the validation dataset, using the code below we'll print 10 predictions vs actual phrase.

{% highlight python %}
batches = [batch for batch in valid_ds]

preds_list = []
ground_truth_list = []

for batch in batches[:1]:
    source = batch[0]
    target = batch[1].numpy()
    bs = tf.shape(source)[0]
    preds = model.generate(source, start_token_idx)
    preds = preds.numpy()

    for i in range(bs):
        target_text = "".join([idx_to_char[_] for _ in target[i, :]])
        ground_truth_list.append(target_text.replace('P', ''))
        prediction = ""
        for idx in preds[i, :]:
            prediction += idx_to_char[idx]
            if idx == end_token_idx:
                break
        preds_list.append(prediction)

for i in range(10):
    print("Target:     ", ground_truth_list[i])
    print("Prediction: ", preds_list[i])
    print('\n~~\n')
{% endhighlight %}  
<br>

At first glance the model seems to be fairly accurate, with mixed performance on different phrases. There were some predictions that were 100% correct, some that were partially correct and some that were completely wrong.  
<br>

![Predictions](/img/posts/ASLFingerspelling/predictions.png)  
<br>

## Results and Future Steps
The model achieved an optimsed training loss of 0.4481 and validation loss of 0.4841, with mean Levenshtein distances of 1.0740 and 1.0737 respectively.  

The yardstick for overall evaluation of the model is the normalised total Levenshtein distance, so let's calculate this.

{% highlight python %}
# Calculate normalised total Levenshtein distance
ground_truth_processed = [ground_truth_list[i][1:-1] for i in range(len(ground_truth_list))]
preds_list_processed = [preds_list[i][1:-1] for i in range(len(preds_list))]
lev_dist = [lev.distance(ground_truth_processed[i], preds_list_processed[i]) 
            for i in range(len(preds_list_processed))]
N = [len(phrase) for phrase in ground_truth_processed]

print('Validation score: '+str((np.sum(N) - np.sum(lev_dist))/np.sum(N)))
{% endhighlight %}  
<br>

The normalised total Levenshtein distance for our final Transformer model was calculated to be **0.63**. To put this into context, as score of 0 would mean every character in every phrase was wrong and a score of 1 would mean every character was correct across every phrase.  

The model performed well on some phrases but poorly on others. Overall the model needs further optimising, a few suggestions to improve the model from here would be:
* Collect and incorporate more training data.
* Use ensemble modelling techniques to generalize the model.
* Experiment with differer quantities of transformer layers, hidden units and attention heads.
* Implement some of the techniques listed earlier to address the character class imbalance.
* Experiment with different variants of transformers.
* Fine-tune a pre-trained model on this task and dataset.
* Learning Rate Scheduling.
* Hyperparameter Tuning.  
<br>


## Conclusion
As I wrap up this project, I am reminded of the power of data science and its potential to tackle real-world challenges. My journey through data preprocessing, model design, training, and evaluation has been both educational and rewarding. While this project may not be groundbreaking, it stands as a testament to my commitment to continuous learning and using data science to make a positive impact.

Feel free to explore the full code and details of this project in my portfolio. If you're interested in learning more or have any questions, I'd love to chat further. Thank you for joining me on this journey into the world of ASL fingerspelling recognition!  
<br>

