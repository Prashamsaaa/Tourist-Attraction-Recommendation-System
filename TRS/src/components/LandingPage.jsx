import React, { useState } from 'react';
import { Search, Calendar, Clock, Users } from 'lucide-react';
import LoginPage from './Login';
import RegisterPage from './Register';

const TravelLanding = () => {
  const [showLogin, setShowLogin] = useState(false);
  const [showRegister, setShowRegister] = useState(false);

  const handleLoginClose = () => setShowLogin(false);
  const handleRegisterClose = () => setShowRegister(false);

  return (
    <div className="min-h-screen bg-gradient-to-b from-green-50 to-blue-50">
      {/* Navigation */}
      <nav className="flex items-center justify-between p-8 max-w-7xl mx-10 fixed top-0 w-full bg-transparent z-10">
        <div className="flex items-center space-x-2">
          <h1 className="text-3xl font-bold">
            <span className="text-gray-800">Let's</span>
            <span className="text-green-600">Wander</span>
          </h1>
        </div>
        <div className="hidden md:flex space-x-4">
          <a href="#" className="text-gray-600 hover:text-gray-800">About</a>
        </div>
        <div className="flex space-x-8 ">
          <button 
            onClick={() => setShowLogin(true)}
            className="px-6 py-2 rounded-full bg-gray-800 text-white hover:bg-gray-700 transition-colors"
          >
            Login
          </button>
          <button 
            onClick={() => setShowRegister(true)}
            className="px-6 py-2 rounded-full bg-green-600 text-white hover:bg-green-500 transition-colors"
          >
            Register
          </button>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="fixed inset-0 overflow-hidden">
        <img 
          src="/background.jpg" 
          alt="background" 
          className="w-full h-full object-cover fixed bg-fixed" // Ensures background is fixed
        />
        <div className="absolute inset-0 bg-black bg-opacity-30">
          <div className="max-w-7xl mx-auto h-full flex flex-col justify-center px-4">
            <h2 className="text-4xl md:text-6xl font-bold text-white mb-4">
              Make Your Hassle-Free Travel Plans Now!
            </h2>
            <h3 className="text-5xl md:text-7xl font-bold text-white mb-8">
              To The World Of An<br />Incredible Vacation.
            </h3>
            
            {/* Search Bar */}
            <div className="bg-white p-4 rounded-lg shadow-lg flex flex-col md:flex-row gap-4">
              <div className="flex-1 flex items-center space-x-2 border-b md:border-b-0 pb-2 md:pb-0">
                <Search className="text-gray-400" />
                <input 
                  type="text" 
                  placeholder="Type Destination"
                  className="w-full outline-none"
                />
              </div>
              
              <button className="w-full md:w-auto bg-green-600 text-white px-8 py-2 rounded-lg hover:bg-green-500 transition-colors">
                Explore Now
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Login Modal */}
      {showLogin && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg w-full max-w-md">
            <LoginPage onClose={handleLoginClose} />
          </div>
        </div>
      )}

      {/* Register Modal */}
      {showRegister && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg w-full max-w-md">
            <RegisterPage onClose={handleRegisterClose} />
          </div>
        </div>
      )}
    </div>
  );
};

export default TravelLanding;