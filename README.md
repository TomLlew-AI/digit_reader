# Digit Reader

<h2>About</h2>
<p>This is a GUI application, that allows the user to draw digits on the canvas, and in return a number representing the model's prediction of the number drawn will show. The model is able to predict numbers with multiple digits.</p>

<hr>

<h2>Libraries Used in the Application and Notebook</h2>
<ul>
  <li><strong>CustomTkinter</strong> - GUI Library used to render the application</li>
  <li><strong>NumPy</strong> - Numerical operations and array manipulation</li>
  <li><strong>Matplotlib</strong> - Visualisation of training results</li>
  <li><strong>OpenCV</strong>, <strong>PIL</strong> - Image manipulation</li>
  <li><strong>Joblib</strong> - Loading and saving models</li>
  <li><strong>Scikit-learn</strong> - Training and preprocessing the model</li>
</ul>

<hr>

<h2>About the Model</h2>
<p>The model used to predict the digits is called K Nearest Neighbours. This is a classification model that uses trained instances to predict the new digits</p>

<h2>Data used</h2>
<p>The data used to train the model is from the mnist dataset</p>

<h2>Preprocessing for Training</h2>
<p>In order to train the model, the image data first had to be formatted so that the model can make good predictions. A pipeline consisting of two components was used to do this</p>
<ol>
  <li>MinMaxScaler - Convert the range of the values of the image from 0-255 to 0-1</li>
  <li>ShadeRemover - A custom component that removes transparent pixels around the digit given a certain threshold of opacity</li>
</ol>

<h2>Model Evaluation</h2>
<p>A combination of metrics were used to evaluate the classification model. The results shown are rounded to 3 significant figures. They are also shown in the externals/model.ipynb notebook</p>
<ul>
  <li>Precision (The ratio of instances predicted positive that were actually correct): 96.6%</li>
  <li>Recall (The ratio of positive instance predicted correctly): 96.6%</li>
  <li>Weighted F1 Score (The harmonic mean of precision and recall): 96.6%</li>
  <li>ROC AUC score (The area under the curve with axes false positives and recall): 99.3%</li>
</ul>

<h2>Use of The Model in The Application</h2>

<p>Upon clicking the <strong>predict</strong> button in the application, the image has to follow a different set of preprocessing sets in order to get an accurate prediction</p>

<ol>
  <li>The pixel values are inverted in the applications, and they are also in the range of 0-255, so the range has to be changed to 0-1, and the values have to be inverted</li>
  <li>Using <pre><code>cv2.connectedComponentsWithStats(image)</code></pre>, we will get the different digits in a list</li>
  <li>Then, reshape each identified digit into 28x28, which is the shape of an image of the mnist dataset</li>
  <li>Finally, flatten each image for prediction</li>
</ol>

<h2>Running the Program</h2>

<p>To run the program, follow these steps:</p>

<ol>
  <li>Download or clone the repository to your local machine.</li>
  <li>Open a terminal and navigate to the project directory using <code>cd &lt;path&gt;</code>, replacing <code>&lt;path&gt;</code> with the location of the repository.</li>
  <li>Ensure Python 3 is installed, then run the following command:</li>
</ol>

<pre><code>python3 main.py</code></pre>

<ol start="4">
  <li>If the program launches successfully, a graphical window will appear displaying a canvas to draw on</li>
</ol>
