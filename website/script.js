document.getElementById('uploadButton').addEventListener('click', async () => {
    const fileInput = document.getElementById('fileInput');
    const resultElement = document.getElementById('result');

    if (fileInput.files.length === 0) {
        resultElement.textContent = 'Please select an image file.';
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://localhost:3050/predict/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        resultElement.textContent = `Predicted Blood Group: ${data.predicted_blood_group}`;
    } catch (error) {
        resultElement.textContent = `Error: ${error.message}`;
    }
});