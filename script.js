document.getElementById('email-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    const emailContent = document.getElementById('email-input').value;
    
    const response = await fetch('/classify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email_content: emailContent })
    });

    const result = await response.json();
    document.getElementById('result').innerText = result.prediction;
});
