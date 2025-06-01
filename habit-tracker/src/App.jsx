import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import HabitCard from './components/HabitCard';
import CalendarView from './components/CalendarView';
import StatsPanel from './components/StatsPanel';
import AddHabitModal from './components/AddHabitModal';
import BadgeSystem from './components/BadgeSystem';
import { Plus, Calendar, BarChart3, Award } from 'lucide-react';
import './App.css';

function App() {
  const [habits, setHabits] = useState([]);
  const [view, setView] = useState('habits'); // 'habits', 'calendar', 'stats', 'badges'
  const [showAddModal, setShowAddModal] = useState(false);

  // Load habits from localStorage on mount
  useEffect(() => {
    const savedHabits = localStorage.getItem('habits');
    if (savedHabits) {
      setHabits(JSON.parse(savedHabits));
    }
  }, []);

  // Save habits to localStorage whenever habits change
  useEffect(() => {
    localStorage.setItem('habits', JSON.stringify(habits));
  }, [habits]);

  const addHabit = (habitData) => {
    const newHabit = {
      id: Date.now(),
      ...habitData,
      streak: 0,
      completedDates: [],
      createdAt: new Date().toISOString()
    };
    setHabits([...habits, newHabit]);
  };

  const toggleHabit = (habitId, date = new Date().toDateString()) => {
    setHabits(habits.map(habit => {
      if (habit.id === habitId) {
        const completedDates = [...habit.completedDates];
        const dateIndex = completedDates.indexOf(date);
        
        if (dateIndex > -1) {
          completedDates.splice(dateIndex, 1);
        } else {
          completedDates.push(date);
        }
        
        // Calculate streak
        const today = new Date();
        let streak = 0;
        for (let i = 0; i < 365; i++) {
          const checkDate = new Date(today);
          checkDate.setDate(today.getDate() - i);
          if (completedDates.includes(checkDate.toDateString())) {
            streak++;
          } else {
            break;
          }
        }
        
        return { ...habit, completedDates, streak };
      }
      return habit;
    }));
  };

  const deleteHabit = (habitId) => {
    setHabits(habits.filter(habit => habit.id !== habitId));
  };

  const navigationItems = [
    { id: 'habits', icon: Plus, label: 'Habits' },
    { id: 'calendar', icon: Calendar, label: 'Calendar' },
    { id: 'stats', icon: BarChart3, label: 'Stats' },
    { id: 'badges', icon: Award, label: 'Badges' }
  ];

  return (
    <div className="app">
      <header className="app-header">
        <motion.h1 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="app-title"
        >
          Habit Tracker
        </motion.h1>
        
        <nav className="app-nav">
          {navigationItems.map(item => {
            const Icon = item.icon;
            return (
              <motion.button
                key={item.id}
                className={`nav-btn ${view === item.id ? 'active' : ''}`}
                onClick={() => setView(item.id)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Icon size={20} />
                <span>{item.label}</span>
              </motion.button>
            );
          })}
        </nav>
      </header>

      <main className="app-main">
        <AnimatePresence mode="wait">
          {view === 'habits' && (
            <motion.div
              key="habits"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="view-container"
            >
              <div className="view-header">
                <h2>Today's Habits</h2>
                <motion.button
                  className="add-btn"
                  onClick={() => setShowAddModal(true)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Plus size={20} />
                  Add Habit
                </motion.button>
              </div>
              
              <div className="habits-grid">
                {habits.map((habit, index) => (
                  <HabitCard
                    key={habit.id}
                    habit={habit}
                    onToggle={toggleHabit}
                    onDelete={deleteHabit}
                    index={index}
                  />
                ))}
              </div>
              
              {habits.length === 0 && (
                <motion.div 
                  className="empty-state"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <p>No habits yet. Create your first habit to get started!</p>
                </motion.div>
              )}
            </motion.div>
          )}
          
          {view === 'calendar' && (
            <motion.div
              key="calendar"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
            >
              <CalendarView habits={habits} onToggle={toggleHabit} />
            </motion.div>
          )}
          
          {view === 'stats' && (
            <motion.div
              key="stats"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
            >
              <StatsPanel habits={habits} />
            </motion.div>
          )}
          
          {view === 'badges' && (
            <motion.div
              key="badges"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
            >
              <BadgeSystem habits={habits} />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <AnimatePresence>
        {showAddModal && (
          <AddHabitModal
            onClose={() => setShowAddModal(false)}
            onAdd={addHabit}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
