import { useState } from "react";
import { ChevronDown } from "lucide-react";

const preferences = [
  {
    name: "category",
    label: "Category",
    options: ["Cultural", "Tourism", "Park", "Nature"],
  },
];

export default function PreferencesPage() {
  const [userPreferences, setUserPreferences] = useState({
    category: "Cultural",
  });

  const handlePreferenceChange = (name, value) => {
    setUserPreferences((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Preferences submitted:", userPreferences);
  };

  return (
    <div className="min-h-screen bg-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md mx-auto bg-white rounded-xl shadow-md overflow-hidden md:max-w-2xl">
        <div className="md:flex">
          <div className="p-8 w-full">
            <div className="uppercase tracking-wide text-sm text-indigo-500 font-semibold mb-1">
              User Preferences
            </div>
            <h1 className="block mt-1 text-lg leading-tight font-medium text-black">
              Customize Your Experience
            </h1>
            <p className="mt-2 text-gray-500">
              Select your preferences to personalize your account.
            </p>
            <form onSubmit={handleSubmit} className="mt-6">
              {preferences.map((pref) => (
                <div key={pref.name} className="mb-4">
                  <label
                    htmlFor={pref.name}
                    className="block text-sm font-medium text-gray-700"
                  >
                    {pref.label}
                  </label>
                  <div className="mt-1 relative">
                    <select
                      id={pref.name}
                      name={pref.name}
                      className="block appearance-none w-full bg-white border border-gray-300 rounded-md py-2 pl-3 pr-10 text-base focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                      value={userPreferences[pref.name]}
                      onChange={(e) =>
                        handlePreferenceChange(pref.name, e.target.value)
                      }
                    >
                      {pref.options.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                    <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                      <ChevronDown className="h-4 w-4" />
                    </div>
                  </div>
                </div>
              ))}
              <div className="mt-6">
                <button
                  type="submit"
                  className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                >
                  Save Preferences
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
