import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, Target, Calendar, Award } from 'lucide-react';

const StatsPanel = ({ habits }) => {
  const today = new Date().toDateString();
  
  const stats = {
    totalHabits: habits.length,
    completedToday: habits.filter(h => h.completedDates.includes(today)).length,
    longestStreak: Math.max(...habits.map(h => h.streak), 0),
    totalCompletions: habits.reduce((sum, h) => sum + h.completedDates.length, 0),
    averageStreak: habits.length > 0 
      ? (habits.reduce((sum, h) => sum + h.streak, 0) / habits.length).toFixed(1)
      : 0
  };
  
  const completionRate = habits.length > 0 
    ? ((stats.completedToday / habits.length) * 100).toFixed(1)
    : 0;

  const statCards = [
    {
      title: 'Completed Today',
      value: `${stats.completedToday}/${stats.totalHabits}`,
      subtitle: `${completionRate}% completion rate`,
      icon: Target,
      color: '#10B981'
    },
    {
      title: 'Longest Streak',
      value: stats.longestStreak,
      subtitle: 'days in a row',
      icon: TrendingUp,
      color: '#F59E0B'
    },
    {
      title: 'Total Completions',
      value: stats.totalCompletions,
      subtitle: 'all time',
      icon: Calendar,
      color: '#3B82F6'
    },
    {
      title: 'Average Streak',
      value: stats.averageStreak,
      subtitle: 'across all habits',
      icon: Award,
      color: '#8B5CF6'
    }
  ];

  return (
    <div className="stats-panel">
      <h2>Your Progress</h2>
      
      <div className="stats-grid">
        {statCards.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <motion.div
              key={stat.title}
              className="stat-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ y: -5, boxShadow: '0 10px 25px rgba(0,0,0,0.1)' }}
            >
              <div className="stat-icon" style={{ color: stat.color }}>
                <Icon size={24} />
              </div>
              
              <div className="stat-content">
                <h3 className="stat-value">{stat.value}</h3>
                <p className="stat-title">{stat.title}</p>
                <p className="stat-subtitle">{stat.subtitle}</p>
              </div>
            </motion.div>
          );
        })}
      </div>
      
      <motion.div 
        className="habits-breakdown"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
      >
        <h3>Habit Breakdown</h3>
        <div className="breakdown-list">
          {habits.map((habit, index) => {
            const completionRate = habit.completedDates.length > 0 
              ? ((habit.completedDates.filter(date => {
                  const completedDate = new Date(date);
                  const daysSinceCreated = Math.floor(
                    (new Date() - new Date(habit.createdAt)) / (1000 * 60 * 60 * 24)
                  );
                  return daysSinceCreated > 0;
                }).length / Math.max(1, Math.floor(
                  (new Date() - new Date(habit.createdAt)) / (1000 * 60 * 60 * 24)
                ))) * 100).toFixed(1)
              : 0;
              
            return (
              <motion.div
                key={habit.id}
                className="breakdown-item"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 + index * 0.1 }}
              >
                <div className="breakdown-info">
                  <span className="habit-name">{habit.name}</span>
                  <span className="habit-stats">
                    {habit.streak} day streak • {completionRate}% completion
                  </span>
                </div>
                
                <div className="progress-bar">
                  <motion.div
                    className="progress-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(completionRate, 100)}%` }}
                    transition={{ duration: 1, delay: 0.7 + index * 0.1 }}
                  />
                </div>
              </motion.div>
            );
          })}
        </div>
      </motion.div>
    </div>
  );
};

export default StatsPanel;