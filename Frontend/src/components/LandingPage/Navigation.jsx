import { useNavigate } from "react-router";

export default function Navigation() {
  const navigate = useNavigate();
  const handleLogin = () => {
    navigate("/login");
  };
  return (
    <nav className="border-b">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex-shrink-0">
            <a href="/" className="text-xl font-semibold">
              TRS.
            </a>
          </div>

          {/* Navigation Links - Hidden on mobile */}
          <div className="hidden md:flex items-center space-x-8">
            <a href="#" className="text-gray-600 hover:text-gray-900">
              Deals
            </a>
            <a href="#" className="text-gray-600 hover:text-gray-900">
              Support
            </a>
            <a href="#" className="text-gray-600 hover:text-gray-900">
              Partnership
            </a>
            <a href="#" className="text-gray-600 hover:text-gray-900">
              Bookings
            </a>
          </div>

          {/* Right side buttons */}
          <div className="flex items-center space-x-4">
            <button
              className="px-4 py-2 text-sm text-gray-700 hover:text-gray-900 rounded-3xl"
              onClick={handleLogin}
            >
              Log in
            </button>
            <button className="px-4 py-2 text-sm text-white bg-indigo-600 hover:bg-indigo-700 rounded-3xl">
              Register
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}
