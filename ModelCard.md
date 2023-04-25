# Model Card for LofiGen
This model is trained on amplitude data from lofi beats and is then tasked with predicting the
next amplitude value given a context of amplitude. Ultimately, once the model is done training,
it is able to generate its own audio and thus new music.

## Model Details

### Model Description

The model is 4-6 layers of 1d convolutional kernels. There are 2 important aspects
that differentiate kernel layers.
The first is channels of each layer. The first half
of the layers gradually increase in channel size. For example, the first 1dConv layer
grows from 1 to 16 channels and the second 1dConv layer grows from 16 to 64 channels.
This gradual increase is very similar to the model in WaveNet where channels are increasing.
The second half of the layers gradually decrease in channel size. For example the last
1dConv layer shrinks from 16 to 1 channel. This is the opposite of the first layer because
allowing for the output dimensions to match the input dimensions.
The second is kernel size. As we go into higher channel spaces, we increase the kernel
size in order to gain more interesting information about our amplitudes over longer
time periods. For example, in our first layer our kernel size is 128, but in our second
layer our kernel size is 256. Similar to the matching increase and decrease in channel size,
for our kernel size the last layer matches the first layer, second layer matches second to last
layer, etc.


- **Developed by:** Suket Shah and Camden Kitowski
- **Model type:** 1D Autoregressive Convolutional Neural Network
- **License:** No License

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** {{ repo | default("[More Information Needed]", true)}}
- **Paper [optional]:** {{ paper | default("[More Information Needed]", true)}}
- **Demo [optional]:** {{ demo | default("[More Information Needed]", true)}}

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->


### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

The primary intended use of LofiGen is to generate new and unique lofi music. This model can be beneficial for music producers looking to create fresh lofi beats, streaming platforms that want to expand their library of lofi music, background music providers for commercial, retail, or public spaces, and creative individuals seeking inspiration for their own musical projects.
The foreseeable users of LofiGen include musicians and music producers, audio engineers and sound designers, streaming platform curators, content creators such as YouTubers and podcasters, and game developers.
While the direct users of LofiGen are those involved in the creation and curation of music, the model can indirectly affect a wider audience. This includes listeners who consume the generated music, artists whose work is being used as training data, and competing music creators who may be influenced by the generated music.


### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->
Downstream uses for the LofiGen model can include providing background music for relaxation or study purposes, enhancing the ambiance in cafes or other public spaces, creating soundtracks for video games, films, or other forms of media, and generating royalty-free music for content creators to use in their videos or podcasts without copyright concerns. Additionally, the generated music can be used for creating music playlists on streaming platforms, further diversifying the available content and offering unique listening experiences to users.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->
Generating music in genres other than lofi: Since LofiGen is specifically trained on lofi beats, it may not be suitable for generating music in other genres or styles, such as classical, rock, or electronic music.

Producing high-quality professional recordings: LofiGen is designed to generate music based on amplitude data, and the resulting audio might not meet the quality standards required for professional music production or commercial releases without further processing and refinement.

Replacing human musicians or composers: LofiGen should not be considered as a replacement for the creativity and expertise of human musicians and composers. Instead, it should be seen as a tool that can inspire and support the creative process.

Mimicking specific artists or songs: The model is not designed to recreate the work of particular artists or to replicate specific songs. Attempting to use LofiGen for this purpose could lead to copyright issues and the unethical use of existing artists' work.

Generating music with lyrics: LofiGen is not designed to generate lyrics or vocal melodies, as it is specifically focused on predicting amplitude values for lofi beats. Using the model to create music with lyrics would be beyond its intended scope.

Real-time live performance: Due to the nature of the model's prediction process and the potential for latency, LofiGen may not be suitable for real-time live performance applications where instantaneous audio generation is required.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->
Some biases risks and limitations with LofiGen include training data bias:

If the training data used to develop the model is not diverse or representative of various lofi styles and subgenres, the generated music might reflect the biases present in the data, leading to a limited range of creative output.

Cultural bias: If the training data is predominantly from one cultural background or region, the generated music may lack the richness and diversity found in lofi music from various cultures and traditions, potentially perpetuating stereotypes or underrepresenting certain styles.

Overfitting and repetitive patterns: If the model overfits the training data, it might produce music that closely resembles the original input, resulting in repetitive patterns or a lack of originality in the generated output.

Copyright infringement risk: There is a possibility that the generated music may closely resemble existing works, leading to potential copyright issues. Users must ensure that the generated music is sufficiently distinct from the training data to avoid legal and ethical concerns.

Quality limitations: The quality of the generated music may be limited by the quality and diversity of the training data, as well as the complexity of the model. The resulting audio may not meet professional standards and may require further processing or refinement.

Unintended content: There is a risk that the model may generate music with elements that are offensive, culturally insensitive, or inappropriate for certain audiences. Users should be cautious when distributing or utilizing the generated content, and take necessary steps to review and curate the output.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Recommendations for Addressing Bias, Risk, and Technical Limitations:

Diverse and representative training data: Ensure that the training data is diverse and representative of various lofi styles, subgenres, and cultural backgrounds. This will help mitigate potential biases and allow the model to generate a broader range of creative output.

Regular model evaluation and updates: Continuously evaluate the model's performance and update the training data to include new lofi music trends and styles. This will help minimize overfitting, encourage originality, and improve the quality of the generated music.

Post-processing and refinement: Users should consider applying post-processing techniques or collaborating with musicians and sound engineers to refine the generated audio, ensuring that it meets quality standards and is suitable for the intended purpose.

Transparency and communication: Clearly communicate to users and affected parties that the generated music is created using an AI model. This will help maintain trust and manage expectations regarding the model's limitations and potential biases.

Copyright-awareness and ethical use: Encourage users to ensure that the generated music is sufficiently distinct from the training data to avoid copyright issues. Emphasize the importance of using the model ethically and respecting the rights of original artists.

Content review and curation: Implement a content review and curation process for the generated music to identify and remove any elements that may be offensive, culturally insensitive, or inappropriate for the intended audience.

Collaborative approach: Encourage users to view LofiGen as a tool that supports and inspires the creative process, rather than as a replacement for human musicians and composers. This will help ensure that the model is used ethically and in a manner that enriches the music industry.

Explore model improvements: Continuously research and develop model improvements to address technical limitations, enhance the quality of the generated music, and expand the model's capabilities to cater to a wider range of use cases and applications.
## How to Get Started with the Model

Use the code below to get started with the model:

https://github.com/CamdenKitowski/Lofi_Generation

## Training Details

### Training Data

Training Card is included in the Github Repo as well.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

The training procedure for LofiGen is a crucial aspect of the model's performance and should be closely linked to the technical specifications. The following points outline the key considerations and steps for the training process, with references to the technical specifications where relevant.

Data collection and preprocessing: Gather a diverse and representative dataset of lofi music, ensuring that it includes various styles, subgenres, and cultural backgrounds. Preprocess the data by extracting amplitude values from the audio files and creating a suitable input format for the model.

Model selection and architecture: Choose an appropriate model architecture based on the task requirements and desired performance. For instance, recurrent neural networks (RNNs) or transformers can be considered, as they have proven effective in sequence-to-sequence learning tasks such as music generation.

Splitting the data: Divide the dataset into training, validation, and test sets. This will allow for effective model training, hyperparameter tuning, and unbiased evaluation of the model's performance.

Model training: Train the model on the prepared dataset, adjusting the learning rate, batch size, and other relevant hyperparameters to optimize the training process. Regularly monitor the model's performance on the validation set to avoid overfitting and to ensure generalization.

Model evaluation: Assess the performance of the trained model on the test set, utilizing appropriate evaluation metrics such as mean squared error (MSE) or mean absolute error (MAE) for amplitude prediction. Consider subjective evaluation methods, such as expert or user feedback, to gauge the perceived quality and creativity of the generated music.

Model iteration and updates: Based on the evaluation results, update the model architecture or training parameters as needed. Continuously refine the model to improve its performance and adapt to new trends and styles in lofi music.

Documentation and transparency: Clearly document the technical specifications, including the model architecture, training data, and hyperparameters, as well as the training and evaluation procedures. This will help users understand the model's capabilities, limitations, and potential biases, ensuring ethical and appropriate usage.

By closely aligning the training procedure with the technical specifications, LofiGen's performance can be optimized, and potential biases, risks, and limitations can be better managed, resulting in a more effective and reliable music generation model.


#### Training Hyperparameters

We trained hyperparameters via manual testing. Mainly, we changed the hyperparameter of
learning rate from .01 all the way down to .0001 which is the current training learning rate.

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->
The evaluation of the LofiGen model is essential for understanding its performance and potential limitations. This section outlines the evaluation protocol focused on the Mean Squared Error (MSE) metric and presents the results obtained.

Quantitative Evaluation using Mean Squared Error (MSE):
MSE is a widely used metric for assessing the accuracy of regression models, such as LofiGen, which predicts amplitude values. MSE measures the average squared difference between the predicted amplitude values and the ground truth values from the test dataset. Lower MSE values indicate better model performance, as the predictions are closer to the actual values.

The evaluation process involves the following steps:

a. Generate amplitude predictions: Use the trained LofiGen model to predict amplitude values for the test dataset.

b. Calculate MSE: Compare the predicted amplitude values with the ground truth values from the test dataset and compute the MSE using the following formula:

MSE = (1/N) * Σ(actual_amplitude - predicted_amplitude)^2

where N is the number of amplitude values in the test dataset, actual_amplitude refers to the ground truth amplitude value, and predicted_amplitude represents the amplitude value predicted by the model.

c. Interpret MSE: Analyze the obtained MSE value to assess the model's performance. Lower MSE values indicate better accuracy in amplitude prediction, while higher MSE values suggest room for improvement.

Results: our final MSE loss for training and validation was effectively 0.

Limitations and Future Work:
Based on the evaluation results using MSE, identify any limitations in the model's performance or areas for improvement. This may include addressing biases in the training data, refining the model architecture, or fine-tuning the model's hyperparameters.

By focusing on the MSE evaluation protocol and analyzing the results, the LofiGen model's performance in amplitude prediction can be better understood. Additionally, this evaluation process can inform future iterations of the model, guiding improvements and addressing any limitations or biases present.

### Testing Data, Factors & Metrics

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->
We did not create any subpopulations or domains. The only factors we tested by was different
genres for experiment 2.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

There were 2 evaluation metrics we used. The first was the MSE Loss which is described above. The second is the auditory output that is generated from our models. This auditory output is subjectively measured while the MSE loss is objective.

### Results

All of our results are outlined in the paper attached with this project.


## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** NVIDIA GeForce RTX 2080 Super GPU
- **Hours used:** 12 hours
- **Cloud Provider:** None. Personal Infrastructure
- **Carbon Emitted:** 1.3 kg C)2

## Model Card Contact

Emails:
Suket Shah: Suket.shah@utexas.edu
Camden Kitowski: camdenkitowski@utexas.edu
