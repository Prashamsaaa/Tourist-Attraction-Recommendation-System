import { useEffect, useState } from "react";
import { ArrowLeft, ChevronDown, LogOut, X } from "lucide-react";
import {
  createPreference,
  getPreferenceByUserId,
} from "../slices/PreferenceSlice";
import { useDispatch, useSelector } from "react-redux";
import DisplayCard from "./Preference/DisplayCard";
import { useNavigate } from "react-router";
import { logout } from "../slices/AuthStoreSlice";

const preferences = [
  {
    name: "category",
    label: "Category",
    options: ["Cultural", "Tourism", "Park", "Nature"],
  },
];

const places = [
  {
    name: "Mountain Retreat",
    image:
      "https://www.holidify.com/images/cmsuploads/compressed/shutterstock_223257994_20190822132504.jpg",
    rating: 4.5,
  },
  {
    name: "Seaside Villa",
    image:
      "https://www.holidify.com/images/cmsuploads/compressed/shutterstock_627150563_20190822130709_20190822154343.jpg",
    rating: 4.8,
  },
  {
    name: "Urban Loft",
    image:
      "https://www.holidify.com/images/cmsuploads/compressed/shutterstock_647026006_20190822122032.jpg",
    rating: 4.2,
  },
  {
    name: "Desert Oasis",
    image: "https://www.holidify.com/images/bgImages/JANAKPUR.jpg",
    rating: 4.6,
  },
];

export default function PreferencesPage() {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { loading, userPreferenceData } = useSelector(
    (state) => state.preference
  );

  const [selectedPreferences, setSelectedPreference] = useState([]);
  const [isPreferenceDropdownOpen, setIsPreferenceDropdownOpen] =
    useState(false);

  const handleBackClick = () => {
    navigate(-1);
  };

  const handleLogout = () => {
    console.log("Logging out...");
    dispatch(logout());
    navigate("/login");
  };

  const handleSelectPreferenceChange = (language) => {
    setSelectedPreference((prev) =>
      prev.includes(language)
        ? prev.filter((lang) => lang !== language)
        : [...prev, language]
    );
  };

  const resetUserPreference = () => {
    setSelectedPreference([]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (selectedPreferences.length) {
      const response = dispatch(createPreference(selectedPreferences));
      console.log("response: ", response);
    }
  };

  useEffect(() => {
    dispatch(getPreferenceByUserId());
  }, [dispatch]);

  return (
    <div className="min-h-screen bg-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6 flex justify-between items-center">
          <button
            onClick={handleBackClick}
            className="flex items-center text-indigo-600 hover:text-indigo-800 transition-colors duration-200"
          >
            <ArrowLeft className="w-5 h-5 mr-2" />
            Back
          </button>
          <button
            onClick={handleLogout}
            className="flex items-center text-red-600 hover:text-red-800 transition-colors duration-200"
          >
            <LogOut className="w-5 h-5 mr-2" />
            Logout
          </button>
        </div>
        <div className="bg-white rounded-xl shadow-md overflow-hidden mb-8 h-[400px]">
          <div className="p-8">
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
              <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                {/* preference multiple select */}
                <div className="sm:col-span-2 mt-6">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Category
                  </label>
                  <div className="relative">
                    <div
                      className="block appearance-none w-full bg-white border border-gray-300 rounded-md py-2 pl-3 pr-10 text-base focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm cursor-pointer"
                      onClick={() =>
                        setIsPreferenceDropdownOpen(!isPreferenceDropdownOpen)
                      }
                    >
                      {selectedPreferences.length > 0
                        ? selectedPreferences.join(", ")
                        : "Select Categories"}
                    </div>
                    <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                      <ChevronDown className="h-4 w-4" />
                    </div>
                    {isPreferenceDropdownOpen && (
                      <div className="absolute z-10 mt-1 w-full bg-white shadow-lg max-h-60 rounded-md py-1 text-base ring-1 ring-black ring-opacity-5 overflow-auto focus:outline-none sm:text-sm">
                        {preferences[0].options.map((language) => (
                          <div
                            key={language}
                            className={`${
                              selectedPreferences.includes(language)
                                ? "bg-indigo-100 text-indigo-900"
                                : "text-gray-900"
                            } cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-indigo-50`}
                            onClick={() =>
                              handleSelectPreferenceChange(language)
                            }
                          >
                            <span
                              className={`${
                                selectedPreferences.includes(language)
                                  ? "font-semibold"
                                  : "font-normal"
                              } block truncate`}
                            >
                              {language}
                            </span>
                            {selectedPreferences.includes(language) && (
                              <span className="absolute inset-y-0 right-0 flex items-center pr-4 text-indigo-600">
                                <svg
                                  className="h-5 w-5"
                                  viewBox="0 0 20 20"
                                  fill="currentColor"
                                  aria-hidden="true"
                                >
                                  <path
                                    fillRule="evenodd"
                                    d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                                    clipRule="evenodd"
                                  />
                                </svg>
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {selectedPreferences.map((lang) => (
                      <span
                        key={lang}
                        className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-100 text-indigo-800"
                      >
                        {lang}
                        <button
                          type="button"
                          onClick={() => handleSelectPreferenceChange(lang)}
                          className="flex-shrink-0 ml-1.5 h-4 w-4 rounded-full inline-flex items-center justify-center text-indigo-400 hover:bg-indigo-200 hover:text-indigo-500 focus:outline-none focus:bg-indigo-500 focus:text-white"
                        >
                          <span className="sr-only">Remove {lang}</span>
                          <X className="h-3 w-3" />
                        </button>
                      </span>
                    ))}
                  </div>
                </div>
                {/* End of multiple preference select option */}
              </div>
              <div className="mt-6 flex gap-2">
                <button
                  type="submit"
                  className={`w-full sm:w-auto flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white 
                    ${
                      loading
                        ? "bg-indigo-400 cursor-not-allowed"
                        : "bg-indigo-600 hover:bg-indigo-700"
                    } focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500`}
                  disabled={loading}
                >
                  {loading ? "Saving..." : "Save Preferences"}
                </button>
                <button
                  className="w-full transition-all duration-75 ease-linear sm:w-auto flex justify-center py-2 px-4 border-2 border-indigo-600 rounded-md shadow-sm text-sm font-medium text-indigo-600 bg-transparent hover:bg-indigo-600 hover:text-white focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                  onClick={resetUserPreference}
                >
                  Reset
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* Cards to display places */}
        {userPreferenceData.preference && (
          <DisplayCard
            userPreferences={userPreferenceData.preference}
            places={places}
          />
        )}
      </div>
    </div>
  );
}
