import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { format, startOfMonth, endOfMonth, eachDayOfInterval, isSameMonth, isSameDay } from 'date-fns';

const CalendarView = ({ habits, onToggle }) => {
  const [currentDate, setCurrentDate] = useState(new Date());
  const [selectedHabit, setSelectedHabit] = useState(habits[0]?.id || null);
  
  const monthStart = startOfMonth(currentDate);
  const monthEnd = endOfMonth(currentDate);
  const days = eachDayOfInterval({ start: monthStart, end: monthEnd });
  
  const selectedHabitData = habits.find(h => h.id === selectedHabit);
  
  const categoryColors = {
    health: '#10B981',
    productivity: '#3B82F6',
    learning: '#8B5CF6',
    fitness: '#EF4444',
    mindfulness: '#F59E0B',
    social: '#EC4899',
    creativity: '#06B6D4',
    other: '#6B7280'
  };

  const navigateMonth = (direction) => {
    const newDate = new Date(currentDate);
    newDate.setMonth(currentDate.getMonth() + direction);
    setCurrentDate(newDate);
  };

  return (
    <div className="calendar-view">
      <div className="calendar-header">
        <div className="month-navigation">
          <motion.button
            onClick={() => navigateMonth(-1)}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <ChevronLeft size={20} />
          </motion.button>
          
          <h2>{format(currentDate, 'MMMM yyyy')}</h2>
          
          <motion.button
            onClick={() => navigateMonth(1)}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <ChevronRight size={20} />
          </motion.button>
        </div>
        
        <div className="habit-selector">
          <select 
            value={selectedHabit || ''} 
            onChange={(e) => setSelectedHabit(Number(e.target.value))}
          >
            {habits.map(habit => (
              <option key={habit.id} value={habit.id}>
                {habit.name}
              </option>
            ))}
          </select>
        </div>
      </div>
      
      {selectedHabitData && (
        <motion.div 
          className="calendar-grid"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <div className="weekdays">
            {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
              <div key={day} className="weekday">{day}</div>
            ))}
          </div>
          
          <div className="days-grid">
            {days.map((day, index) => {
              const dayString = day.toDateString();
              const isCompleted = selectedHabitData.completedDates.includes(dayString);
              const isToday = isSameDay(day, new Date());
              
              return (
                <motion.div
                  key={day.toISOString()}
                  className={`calendar-day ${
                    isCompleted ? 'completed' : ''
                  } ${isToday ? 'today' : ''}`}
                  onClick={() => onToggle(selectedHabit, dayString)}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.01 }}
                  style={{
                    backgroundColor: isCompleted 
                      ? categoryColors[selectedHabitData.category] 
                      : 'transparent'
                  }}
                >
                  <span>{format(day, 'd')}</span>
                </motion.div>
              );
            })}
          </div>
        </motion.div>
      )}
      
      {habits.length === 0 && (
        <div className="empty-calendar">
          <p>No habits to display. Create a habit first!</p>
        </div>
      )}
    </div>
  );
};

export default CalendarView;