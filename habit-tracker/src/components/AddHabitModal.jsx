import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { X } from 'lucide-react';

const AddHabitModal = ({ onClose, onAdd }) => {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    category: 'health'
  });

  const categories = [
    { value: 'health', label: 'Health', color: '#10B981' },
    { value: 'productivity', label: 'Productivity', color: '#3B82F6' },
    { value: 'learning', label: 'Learning', color: '#8B5CF6' },
    { value: 'fitness', label: 'Fitness', color: '#EF4444' },
    { value: 'mindfulness', label: 'Mindfulness', color: '#F59E0B' },
    { value: 'social', label: 'Social', color: '#EC4899' },
    { value: 'creativity', label: 'Creativity', color: '#06B6D4' },
    { value: 'other', label: 'Other', color: '#6B7280' }
  ];

  const handleSubmit = (e) => {
    e.preventDefault();
    if (formData.name.trim()) {
      onAdd(formData);
      onClose();
    }
  };

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <motion.div
      className="modal-overlay"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
    >
      <motion.div
        className="modal-content"
        initial={{ opacity: 0, scale: 0.8, y: 50 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.8, y: 50 }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h2>Add New Habit</h2>
          <button className="close-btn" onClick={onClose}>
            <X size={20} />
          </button>
        </div>
        
        <form onSubmit={handleSubmit} className="habit-form">
          <div className="form-group">
            <label htmlFor="name">Habit Name</label>
            <input
              id="name"
              type="text"
              value={formData.name}
              onChange={(e) => handleChange('name', e.target.value)}
              placeholder="e.g., Drink 8 glasses of water"
              required
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="description">Description (optional)</label>
            <textarea
              id="description"
              value={formData.description}
              onChange={(e) => handleChange('description', e.target.value)}
              placeholder="Why is this habit important to you?"
              rows={3}
            />
          </div>
          
          <div className="form-group">
            <label>Category</label>
            <div className="category-grid">
              {categories.map(category => (
                <motion.button
                  key={category.value}
                  type="button"
                  className={`category-btn ${
                    formData.category === category.value ? 'selected' : ''
                  }`}
                  onClick={() => handleChange('category', category.value)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  style={{
                    borderColor: category.color,
                    backgroundColor: formData.category === category.value 
                      ? category.color 
                      : 'transparent',
                    color: formData.category === category.value 
                      ? '#FFFFFF' 
                      : category.color
                  }}
                >
                  {category.label}
                </motion.button>
              ))}
            </div>
          </div>
          
          <div className="form-actions">
            <button type="button" onClick={onClose} className="cancel-btn">
              Cancel
            </button>
            <motion.button
              type="submit"
              className="submit-btn"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Create Habit
            </motion.button>
          </div>
        </form>
      </motion.div>
    </motion.div>
  );
};

export default AddHabitModal;