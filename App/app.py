import gradio as gr
import skops.io as sio

untrusted_types = sio.get_untrusted_types(
    file="Model/iris_pipeline.skops"
)
pipe = sio.load(
    "Model/iris_pipeline.skops", trusted=untrusted_types
)

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    """
    Predict Iris species based on flower measurements.
    
    Args:
        sepal_length (float): Sepal length in cm
        sepal_width (float): Sepal width in cm
        petal_length (float): Petal length in cm
        petal_width (float): Petal width in cm
    
    Returns:
        str: Predicted species
    """
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = pipe.predict(features)[0]
    
    species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    species = species_map[prediction]
    
    return f"🌸 Predicted Species: **{species}**"

# Define inputs
inputs = [
    gr.Slider(4.0, 8.0, step=0.1, label="Sepal Length (cm)", value=5.1),
    gr.Slider(2.0, 4.5, step=0.1, label="Sepal Width (cm)", value=3.5),
    gr.Slider(1.0, 7.0, step=0.1, label="Petal Length (cm)", value=1.4),
    gr.Slider(0.1, 2.5, step=0.1, label="Petal Width (cm)", value=0.2),
]

# Example inputs
examples = [
    [5.1, 3.5, 1.4, 0.2],  # Setosa
    [6.7, 3.1, 4.7, 1.5],  # Versicolor
    [6.3, 2.9, 5.6, 1.8],  # Virginica
]

# Create interface
demo = gr.Interface(
    fn=predict_iris,
    inputs=inputs,
    outputs=gr.Markdown(),
    examples=examples,
    title="🌺 Iris Species Classifier",
    description="Enter flower measurements to predict the Iris species (Setosa, Versicolor, or Virginica)",
    article="This app uses a Random Forest model trained with CI/CD automation via GitHub Actions.",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()