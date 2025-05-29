// Timer state
let isRunning = false;
let currentTime = 1500; // This will be updated after settings load
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
    const weekNumber = Math.ceil((((date - firstDayOfYear) / 86400000) + firstDayOfYear.getDay() + 1) / 7);
    // Return as string with year prefix to ensure uniqueness across years
    return `${date.getFullYear()}-W${weekNumber.toString().padStart(2, '0')}`;
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

// Add chart instances tracking
let dailyChart = null;
let weeklyChart = null;
let tasksChart = null;

function renderCharts() {
    // Destroy existing charts before creating new ones
    if (dailyChart) {
        dailyChart.destroy();
    }
    if (weeklyChart) {
        weeklyChart.destroy();
    }
    if (tasksChart) {
        tasksChart.destroy();
    }

    // Daily Chart
    const dailyCtx = document.getElementById('dailyChart').getContext('2d');
    dailyChart = new Chart(dailyCtx, {
        type: 'bar',
        data: {
            labels: Object.keys(statistics.daily).slice(-7),
            datasets: [{
                label: 'Focus Time (minutes)',
                data: Object.values(statistics.daily).slice(-7).map(day => day?.focusTime || 0),
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
    weeklyChart = new Chart(weeklyCtx, {
        type: 'line',
        data: {
            labels: Object.keys(statistics.weekly).slice(-4).map(week => {
                // Convert week key back to readable format
                if (week.includes('-W')) {
                    return week; // Already in new format
                }
                return `Week ${week}`; // Legacy format
            }),
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
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });

    // Tasks Completed Chart
    const tasksCtx = document.getElementById('tasksChart').getContext('2d');
    tasksChart = new Chart(tasksCtx, {
        type: 'bar',
        data: {
            labels: Object.keys(statistics.daily).slice(-7),
            datasets: [{
                label: 'Tasks Completed',
                data: Object.values(statistics.daily).slice(-7).map(day => day.tasksCompleted),
                backgroundColor: 'rgba(255, 159, 64, 0.2)',
                borderColor: 'rgba(255, 159, 64, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
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
        
        // Initialize daily statistics if not exists
        if (!statistics.daily[today]) {
            statistics.daily[today] = {
                focusTime: 0,
                tasksCompleted: 0,
                pomodoros: 0
            };
        }
        
        // Initialize weekly statistics if not exists
        if (!statistics.weekly[currentWeek]) {
            statistics.weekly[currentWeek] = {
                focusTime: 0,
                tasksCompleted: 0,
                pomodoros: 0
            };
        }
        
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

function deleteTask(taskId) {
    tasks = tasks.filter(task => task.id !== taskId);
    saveTasks();
    renderTasks();
    
    // Update statistics
    const today = new Date().toLocaleDateString();
    const currentWeek = getWeekNumber(new Date());
    
    // Initialize statistics for today if not exists
    if (!statistics.daily[today]) {
        statistics.daily[today] = {
            focusTime: 0,
            tasksCompleted: 0,
            pomodoros: 0
        };
    }
    
    // Initialize statistics for current week if not exists
    if (!statistics.weekly[currentWeek]) {
        statistics.weekly[currentWeek] = {
            focusTime: 0,
            tasksCompleted: 0,
            pomodoros: 0
        };
    }
    
    // Update both daily and weekly statistics
    statistics.daily[today].tasksCompleted += 1;
    statistics.weekly[currentWeek].tasksCompleted += 1;
    
    // Save statistics and update display
    localStorage.setItem('statistics', JSON.stringify(statistics));
    renderCharts();
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


// Settings Management
function loadSettings() {
    const savedSettings = localStorage.getItem('pomodoroSettings');
    if (savedSettings) {
        Object.assign(settings, JSON.parse(savedSettings));
        updateSettingsUI();
    }
    
    // Update timer to reflect current settings
    if (!isRunning) {
        currentTime = settings.workTime * 60;
        updateTimer();
    }
}

function saveSettings() {
    localStorage.setItem('pomodoroSettings', JSON.stringify(settings));
}

function updateSettingsUI() {
    document.getElementById('workTime').value = settings.workTime;
    document.getElementById('shortBreak').value = settings.shortBreak;
    document.getElementById('longBreak').value = settings.longBreak;
    document.getElementById('volume').value = settings.volume;
    
    // Update theme
    document.body.setAttribute('data-theme', settings.theme);
}

function handleSettingChange(event) {
    const setting = event.target.id;
    const value = event.target.value;
    
    // Validate input
    switch(setting) {
        case 'workTime':
            settings.workTime = Math.min(Math.max(parseInt(value) || 25, 1), 60);
            break;
        case 'shortBreak':
            settings.shortBreak = Math.min(Math.max(parseInt(value) || 5, 1), 30);
            break;
        case 'longBreak':
            settings.longBreak = Math.min(Math.max(parseInt(value) || 15, 1), 60);
            break;
        case 'volume':
            settings.volume = Math.min(Math.max(parseInt(value) || 50, 0), 100);
            break;
    }
    
    saveSettings();
    
    // Reset timer if it's not running
    if (!isRunning) {
        resetTimer();
    }
}

// Add event listeners for settings
document.getElementById('workTime').addEventListener('change', handleSettingChange);
document.getElementById('shortBreak').addEventListener('change', handleSettingChange);
document.getElementById('longBreak').addEventListener('change', handleSettingChange);
document.getElementById('volume').addEventListener('change', handleSettingChange);

// Theme switching
document.getElementById('themeToggle').addEventListener('click', () => {
    settings.theme = settings.theme === 'light' ? 'dark' : 'light';
    document.body.setAttribute('data-theme', settings.theme);
    saveSettings();
});

// Initialize settings on load
loadSettings();

// Update timer display after settings are loaded
resetTimer();


// Keyboard Shortcuts
const shortcuts = {
    'Space': () => {
        if (isRunning) {
            pauseTimer();
        } else {
            startTimer();
        }
    },
    'r': resetTimer,
    'n': () => {
        document.getElementById('taskInput').focus();
    },
    'Escape': () => {
        document.getElementById('taskInput').blur();
    },
    't': () => {
        document.getElementById('themeToggle').click();
    },
    '1': () => {
        document.getElementById('workTime').focus();
    },
    '2': () => {
        document.getElementById('shortBreak').focus();
    },
    '3': () => {
        document.getElementById('longBreak').focus();
    }
};

// Add keyboard event listener
document.addEventListener('keydown', (event) => {
    // Ignore shortcuts when typing in input fields
    if (event.target.tagName === 'INPUT') {
        return;
    }
    
    const key = event.key;
    if (shortcuts.hasOwnProperty(key)) {
        event.preventDefault();
        shortcuts[key]();
    }
});

// Add shortcut hints to UI
function addShortcutHints() {
    // Start/Pause button
    startBtn.setAttribute('title', 'Space: Start/Pause timer');
    pauseBtn.setAttribute('title', 'Space: Start/Pause timer');
    
    // Reset button
    resetBtn.setAttribute('title', 'R: Reset timer');
    
    // Task input
    taskInput.setAttribute('title', 'N: Focus task input, Escape: Blur');
    
    // Theme toggle
    document.getElementById('themeToggle')
        .setAttribute('title', 'T: Toggle theme');
    
    // Time settings
    document.getElementById('workTime')
        .setAttribute('title', '1: Focus work time');
    document.getElementById('shortBreak')
        .setAttribute('title', '2: Focus short break');
    document.getElementById('longBreak')
        .setAttribute('title', '3: Focus long break');
}

// Initialize shortcut hints
addShortcutHints();