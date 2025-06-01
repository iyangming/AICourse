import React from 'react';
import { motion } from 'framer-motion';
import { Award, Star, Zap, Target, Calendar, Trophy } from 'lucide-react';

const BadgeSystem = ({ habits }) => {
  const badges = [
    {
      id: 'first-habit',
      name: 'Getting Started',
      description: 'Create your first habit',
      icon: Star,
      condition: () => habits.length >= 1,
      color: '#F59E0B'
    },
    {
      id: 'week-warrior',
      name: 'Week Warrior',
      description: 'Maintain a 7-day streak',
      icon: Zap,
      condition: () => habits.some(h => h.streak >= 7),
      color: '#10B981'
    },
    {
      id: 'month-master',
      name: 'Month Master',
      description: 'Maintain a 30-day streak',
      icon: Calendar,
      condition: () => habits.some(h => h.streak >= 30),
      color: '#3B82F6'
    },
    {
      id: 'habit-collector',
      name: 'Habit Collector',
      description: 'Create 5 different habits',
      icon: Target,
      condition: () => habits.length >= 5,
      color: '#8B5CF6'
    },
    {
      id: 'perfectionist',
      name: 'Perfectionist',
      description: 'Complete all habits for a day',
      icon: Award,
      condition: () => {
        const today = new Date().toDateString();
        return habits.length > 0 && habits.every(h => h.completedDates.includes(today));
      },
      color: '#EC4899'
    },
    {
      id: 'century-club',
      name: 'Century Club',
      description: 'Reach 100 total completions',
      icon: Trophy,
      condition: () => {
        const totalCompletions = habits.reduce((sum, h) => sum + h.completedDates.length, 0);
        return totalCompletions >= 100;
      },
      color: '#EF4444'
    }
  ];

  const earnedBadges = badges.filter(badge => badge.condition());
  const availableBadges = badges.filter(badge => !badge.condition());

  return (
    <div className="badge-system">
      <h2>Achievements</h2>
      
      <div className="badges-section">
        <h3>Earned Badges ({earnedBadges.length})</h3>
        <div className="badges-grid">
          {earnedBadges.map((badge, index) => {
            const Icon = badge.icon;
            return (
              <motion.div
                key={badge.id}
                className="badge earned"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ scale: 1.05, y: -5 }}
              >
                <motion.div
                  className="badge-icon"
                  style={{ color: badge.color }}
                  animate={{ rotate: [0, 10, -10, 0] }}
                  transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
                >
                  <Icon size={32} />
                </motion.div>
                
                <h4 className="badge-name">{badge.name}</h4>
                <p className="badge-description">{badge.description}</p>
                
                <motion.div
                  className="badge-glow"
                  animate={{ opacity: [0.5, 1, 0.5] }}
                  transition={{ duration: 2, repeat: Infinity }}
                  style={{ backgroundColor: badge.color }}
                />
              </motion.div>
            );
          })}
        </div>
        
        {earnedBadges.length === 0 && (
          <p className="no-badges">No badges earned yet. Keep building those habits!</p>
        )}
      </div>
      
      <div className="badges-section">
        <h3>Available Badges</h3>
        <div className="badges-grid">
          {availableBadges.map((badge, index) => {
            const Icon = badge.icon;
            return (
              <motion.div
                key={badge.id}
                className="badge available"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ scale: 1.02 }}
              >
                <div className="badge-icon locked">
                  <Icon size={32} />
                </div>
                
                <h4 className="badge-name">{badge.name}</h4>
                <p className="badge-description">{badge.description}</p>
              </motion.div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default BadgeSystem;