document.addEventListener('DOMContentLoaded', function () {
  const forms = document.querySelectorAll('form');
  forms.forEach(form => {
    form.addEventListener('submit', function (event) {
      if (!form.checkValidity()) {
        event.preventDefault();
        event.stopPropagation();
        form.classList.add('was-validated');
      }
    }, false);
  });
});

// Chatbot Functions
function toggleChatbox () {
  const chatbox = document.querySelector('.chatbox');
  const chatboxToggle = document.querySelector('#chatbox-toggle');
  if (chatbox && chatboxToggle) {
    if (chatbox.style.display === 'none' || chatbox.style.display === '') {
      chatbox.style.display = 'block';
      chatboxToggle.style.display = 'none';
    } else {
      chatbox.style.display = 'none';
      chatboxToggle.style.display = 'block';
    }
  }
}

function sendMessage () {
  const userInput = document.querySelector('#user-input');
  const message = userInput ? userInput.value.trim() : '';
  if (message) {
    appendMessage('User', message);
    if (userInput) userInput.value = '';
    getBotResponse(message);
  }
}

function appendMessage (sender, message) {
  const chatboxContent = document.querySelector('#chatbox-content');
  if (chatboxContent) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender.toLowerCase());
    messageElement.innerText = `${sender}: ${message}`;
    chatboxContent.appendChild(messageElement);
    chatboxContent.scrollTop = chatboxContent.scrollHeight;
  }
}

function getBotResponse (message) {
  // Normalize the message to lowercase for easier matching
  const userMessage = message.toLowerCase().trim();

  // Define responses for health and heart disease-related queries
  let response;

  if (userMessage.includes('hello') || userMessage.includes('hi') || userMessage.includes('hey')) {
    response = 'Hello! I am here to assist you with heart disease information. How can I help you today?';
  } else if (userMessage.includes('heart disease')) {
    response = 'Heart disease refers to a range of conditions that affect the heart, including coronary artery disease, heart attacks, heart failure, and arrhythmias. Would you like to know more about any specific type?';
  } else if (userMessage.includes('risk factors')) {
    response = 'Common risk factors for heart disease include high blood pressure, high cholesterol, smoking, obesity, physical inactivity, and a family history of heart disease. Would you like to know how to reduce these risks?';
  } else if (userMessage.includes('symptoms')) {
    response = 'Symptoms of heart disease may include chest pain, shortness of breath, dizziness, palpitations, or fatigue. If you experience these symptoms, it is important to seek medical advice immediately.';
  } else if (userMessage.includes('prevent') || userMessage.includes('prevention')) {
    response = 'To prevent heart disease, it’s important to maintain a healthy diet, exercise regularly, avoid smoking, and manage stress. Do you need tips on how to get started?';
  } else if (userMessage.includes('cholesterol')) {
    response = 'High cholesterol is a major risk factor for heart disease. You can manage it through diet, exercise, and medication. Would you like more information on how to control cholesterol levels?';
  } else if (userMessage.includes('blood pressure') || userMessage.includes('high blood pressure')) {
    response = 'High blood pressure is a key risk factor for heart disease. You can lower blood pressure through exercise, diet, and medication. Would you like tips on managing blood pressure?';
  } else if (userMessage.includes('exercise')) {
    response = 'Regular exercise is one of the best ways to keep your heart healthy. Activities like walking, running, swimming, and cycling can help reduce the risk of heart disease. Would you like a personalized exercise plan?';
  } else if (userMessage.includes('smoking')) {
    response = 'Smoking is a leading cause of heart disease. Quitting smoking can significantly reduce your risk. Need tips on how to quit?';
  } else if (userMessage.includes('diet')) {
    response = 'A heart-healthy diet includes plenty of fruits, vegetables, whole grains, lean proteins, and healthy fats. Would you like some meal planning ideas or recipes for heart health?';
  } else if (userMessage.includes('heart attack')) {
    response = 'A heart attack occurs when blood flow to a part of the heart muscle is blocked. Common symptoms include chest pain, shortness of breath, and nausea. If you suspect you are having a heart attack, seek emergency medical help immediately.';
  } else if (userMessage.includes('stroke')) {
    response = 'A stroke happens when blood flow to a part of the brain is interrupted. Symptoms include sudden numbness, confusion, trouble speaking, or difficulty walking. Seek medical help if you experience these symptoms.';
  } else if (userMessage.includes('arrhythmia')) {
    response = 'An arrhythmia is an irregular heartbeat, which can lead to complications like stroke or heart failure. If you have symptoms like a racing heart or palpitations, it’s important to get checked by a doctor.';
  } else if (userMessage.includes('heart failure')) {
    response = 'Heart failure occurs when the heart is unable to pump blood efficiently. Symptoms include shortness of breath, fatigue, and swelling in the legs. Would you like to know more about managing heart failure?';
  } else if (userMessage.includes('family history')) {
    response = 'A family history of heart disease increases your risk. It’s important to monitor your heart health and follow a healthy lifestyle. Would you like advice on how to manage your heart health?';
  } else if (userMessage.includes('age') || userMessage.includes('older age')) {
    response = 'As you age, your risk for heart disease increases. However, maintaining a healthy lifestyle can help reduce this risk significantly. Do you want to know more about age-related heart disease risks?';
  } else if (userMessage.includes('mental health')) {
    response = 'Mental health is closely linked to heart health. Stress, anxiety, and depression can contribute to heart disease. Would you like tips on managing stress or improving mental well-being?';
  } else if (userMessage.includes('treatment') || userMessage.includes('medication')) {
    response = 'Treatment for heart disease depends on the type and severity. It may involve lifestyle changes, medications, or surgical procedures. Would you like to learn more about treatments for specific heart conditions?';
  } else if (userMessage.includes('help')) {
    response = 'I can assist you with information about heart disease, symptoms, risk factors, prevention, and treatment. Ask me anything related to heart health!';
  } else if (userMessage.includes('contact')) {
    response = 'You can contact me via email at anurajvenkatpurwar@gmail.com. Feel free to reach out if you have any more questions!';
  } else {
    response = 'Sorry, I didn\'t understand that. Could you rephrase or ask a more specific question about heart health?';
  }

  // Append the bot's response to the chat
  appendMessage('Bot', response);
}
