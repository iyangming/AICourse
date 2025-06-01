import React from 'react';
import { motion } from 'framer-motion';
import { Check, Flame, Trash2 } from 'lucide-react';

const HabitCard = ({ habit, onToggle, onDelete, index }) => {
  const today = new Date().toDateString();
  const isCompletedToday = habit.completedDates.includes(today);
  
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

  return (
    <motion.div
      className="habit-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      whileHover={{ y: -5, boxShadow: '0 10px 25px rgba(0,0,0,0.1)' }}
      style={{ borderLeft: `4px solid ${categoryColors[habit.category]}` }}
    >
      <div className="habit-header">
        <h3 className="habit-name">{habit.name}</h3>
        <button
          className="delete-btn"
          onClick={() => onDelete(habit.id)}
        >
          <Trash2 size={16} />
        </button>
      </div>
      
      <p className="habit-description">{habit.description}</p>
      
      <div className="habit-stats">
        <div className="streak">
          <Flame size={16} color="#F59E0B" />
          <span>{habit.streak} day streak</span>
        </div>
        
        <div className="category">
          <span 
            className="category-badge"
            style={{ backgroundColor: categoryColors[habit.category] }}
          >
            {habit.category}
          </span>
        </div>
      </div>
      
      <motion.button
        className={`complete-btn ${isCompletedToday ? 'completed' : ''}`}
        onClick={() => onToggle(habit.id)}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        animate={{
          backgroundColor: isCompletedToday ? '#10B981' : '#F3F4F6',
          color: isCompletedToday ? '#FFFFFF' : '#6B7280'
        }}
      >
        <motion.div
          animate={{ rotate: isCompletedToday ? 360 : 0 }}
          transition={{ duration: 0.3 }}
        >
          <Check size={20} />
        </motion.div>
        {isCompletedToday ? 'Completed' : 'Mark Complete'}
      </motion.button>
    </motion.div>
  );
};

export default HabitCard;