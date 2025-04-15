function analyzeSentence() {
    const sentence = document.getElementById('sentence').value;

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ sentence: sentence }),
    })
    .then(response => {
        if (!response.ok) {
            return response.text().then(text => {
                throw new Error(`Server error: ${text}`);
            });
        }
        return response.json();
    })
    .then(data => {
        // Update the <p id="result"> element with the prediction
        const resultElement = document.getElementById('result');
        resultElement.textContent = `التنبؤ: ${data.prediction}`; 
    })
    .catch(error => {
        console.error('Error:', error);
        const resultElement = document.getElementById('result');
        resultElement.textContent = 'حدث خطأ أثناء التحليل. الرجاء المحاولة مرة أخرى.';
    });
}