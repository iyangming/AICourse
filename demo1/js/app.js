// Timer state
let isRunning = false;
let currentTime = 1500; // 25 minutes in seconds
let interval = null;
let currentMode = 'work';

// Settings
const settings = {
    workTime: 25,
    shortBreak: 5,
    longBreak: 15,
    volume: 50,
    theme: 'light'
};

// DOM Elements
const minutesDisplay = document.getElementById('minutes');
const secondsDisplay = document.getElementById('seconds');
const startBtn = document.getElementById('startBtn');
const pauseBtn = document.getElementById('pauseBtn');
const resetBtn = document.getElementById('resetBtn');
const progressRing = document.querySelector('.progress-ring-circle');

// Calculate progress ring properties
const radius = progressRing.r.baseVal.value;
const circumference = radius * 2 * Math.PI;
progressRing.style.strokeDasharray = `${circumference} ${circumference}`;
progressRing.style.strokeDashoffset = circumference;

// Timer functions
function updateTimer() {
    const minutes = Math.floor(currentTime / 60);
    const seconds = currentTime % 60;
    
    minutesDisplay.textContent = minutes.toString().padStart(2, '0');
    secondsDisplay.textContent = seconds.toString().padStart(2, '0');
    
    updateProgress();
}

function updateProgress() {
    const totalTime = settings[currentMode === 'work' ? 'workTime' : 'shortBreak'] * 60;
    const progress = currentTime / totalTime;
    const offset = circumference - (progress * circumference);
    progressRing.style.strokeDashoffset = offset;
}

function startTimer() {
    if (!isRunning) {
        isRunning = true;
        interval = setInterval(() => {
            currentTime--;
            if (currentTime < 0) {
                handleTimerComplete();
            } else {
                updateTimer();
            }
        }, 1000);
    }
}

function pauseTimer() {
    isRunning = false;
    clearInterval(interval);
}

function resetTimer() {
    isRunning = false;
    clearInterval(interval);
    currentTime = settings.workTime * 60;
    updateTimer();
}

// Statistics tracking
let statistics = JSON.parse(localStorage.getItem('statistics')) || {
    daily: {},
    weekly: {},
    currentDay: new Date().toLocaleDateString(),
    currentWeek: getWeekNumber(new Date())
};

function getWeekNumber(date) {
    const firstDayOfYear = new Date(date.getFullYear(), 0, 1);
    return Math.ceil((((date - firstDayOfYear) / 86400000) + firstDayOfYear.getDay() + 1) / 7);
}

function updateStatistics() {
    const today = new Date().toLocaleDateString();
    const currentWeek = getWeekNumber(new Date());
    
    // Update daily statistics
    if (!statistics.daily[today]) {
        statistics.daily[today] = {
            focusTime: 0,
            tasksCompleted: 0,
            pomodoros: 0
        };
    }
    
    // Update weekly statistics
    if (!statistics.weekly[currentWeek]) {
        statistics.weekly[currentWeek] = {
            focusTime: 0,
            tasksCompleted: 0,
            pomodoros: 0
        };
    }
    
    localStorage.setItem('statistics', JSON.stringify(statistics));
    renderCharts();
}

function renderCharts() {
    // Daily Chart
    const dailyCtx = document.getElementById('dailyChart').getContext('2d');
    new Chart(dailyCtx, {
        type: 'bar',
        data: {
            labels: Object.keys(statistics.daily).slice(-7),
            datasets: [{
                label: 'Focus Time (minutes)',
                data: Object.values(statistics.daily).slice(-7).map(day => day.focusTime),
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    // Weekly Chart
    const weeklyCtx = document.getElementById('weeklyChart').getContext('2d');
    new Chart(weeklyCtx, {
        type: 'line',
        data: {
            labels: Object.keys(statistics.weekly).slice(-4),
            datasets: [{
                label: 'Weekly Pomodoros',
                data: Object.values(statistics.weekly).slice(-4).map(week => week.pomodoros),
                fill: false,
                borderColor: 'rgba(153, 102, 255, 1)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    // Update summary statistics
    const today = new Date().toLocaleDateString();
    document.getElementById('todayFocus').textContent = 
        `${statistics.daily[today]?.focusTime || 0} minutes`;
    document.getElementById('tasksCompleted').textContent = 
        statistics.daily[today]?.tasksCompleted || 0;
}

// Update handleTimerComplete function to track statistics
function handleTimerComplete() {
    playNotification();
    
    if (currentMode === 'work') {
        // Update statistics
        const today = new Date().toLocaleDateString();
        const currentWeek = getWeekNumber(new Date());
        
        statistics.daily[today].focusTime += settings.workTime;
        statistics.daily[today].pomodoros += 1;
        statistics.weekly[currentWeek].pomodoros += 1;
        statistics.weekly[currentWeek].focusTime += settings.workTime;
        
        localStorage.setItem('statistics', JSON.stringify(statistics));
        renderCharts();
        
        currentMode = 'break';
        currentTime = settings.shortBreak * 60;
    } else {
        currentMode = 'work';
        currentTime = settings.workTime * 60;
    }
    
    updateTimer();
    pauseTimer();
}

// Initialize statistics
updateStatistics();

// Task management
const taskInput = document.getElementById('taskInput');
const addTaskBtn = document.getElementById('addTaskBtn');
const taskList = document.getElementById('taskList');

let tasks = JSON.parse(localStorage.getItem('tasks')) || [];

function addTask(taskText) {
    const task = {
        id: Date.now(),
        text: taskText,
        completed: false,
        pomodoros: 0
    };
    tasks.push(task);
    saveTasks();
    renderTasks();
}

function saveTasks() {
    localStorage.setItem('tasks', JSON.stringify(tasks));
}

function renderTasks() {
    taskList.innerHTML = '';
    tasks.forEach(task => {
        const li = document.createElement('li');
        li.innerHTML = `
            <span>${task.text} (${task.pomodoros} pomodoros)</span>
            <button onclick="deleteTask(${task.id})">Delete</button>
        `;
        taskList.appendChild(li);
    });
}

// Event listeners
startBtn.addEventListener('click', startTimer);
pauseBtn.addEventListener('click', pauseTimer);
resetBtn.addEventListener('click', resetTimer);

addTaskBtn.addEventListener('click', () => {
    const text = taskInput.value.trim();
    if (text) {
        addTask(text);
        taskInput.value = '';
    }
});

// Initialize
updateTimer();
renderTasks();

// Service Worker Registration
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('ServiceWorker registration successful');
            })
            .catch(err => {
                console.log('ServiceWorker registration failed:', err);
            });
    });
}