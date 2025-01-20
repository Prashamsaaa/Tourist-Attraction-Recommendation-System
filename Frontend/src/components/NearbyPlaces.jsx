import { useState } from "react";
import { ArrowLeft, LogOut, Star } from "lucide-react";
import { useNavigate } from "react-router";
import Map from "./Map";
import { useDispatch } from "react-redux";
import { logout } from "../slices/AuthStoreSlice";

const places = [
  {
    id: 1,
    name: "Mountain Retreat",
    image:
      "https://www.holidify.com/images/cmsuploads/compressed/shutterstock_223257994_20190822132504.jpg",
    rating: 4.5,
    description: "A serene mountain getaway with breathtaking views.",
    x: 20,
    y: 30,
    position: { lat: 27.7172, lng: 85.324 },
  },
  {
    id: 2,
    name: "Seaside Villa",
    image:
      "https://www.holidify.com/images/cmsuploads/compressed/shutterstock_627150563_20190822130709_20190822154343.jpg",
    rating: 4.8,
    description: "Luxurious villa right on the beach with private access.",
    x: 70,
    y: 60,
    position: { lat: 28.2096, lng: 83.9856 },
  },
  {
    id: 3,
    name: "Urban Loft",
    image:
      "https://www.holidify.com/images/cmsuploads/compressed/shutterstock_647026006_20190822122032.jpg",
    rating: 4.2,
    description: "Modern loft in the heart of the city, close to attractions.",
    x: 40,
    y: 50,
    position: { lat: 27.6792, lng: 83.507 },
  },
  {
    id: 4,
    name: "Desert Oasis",
    image: "https://www.holidify.com/images/bgImages/JANAKPUR.jpg",
    rating: 4.6,
    description: "Unique desert retreat with stunning sunset views.",
    x: 80,
    y: 20,
    position: { lat: 27.5782, lng: 83.207 },
  },
  {
    id: 5,
    name: "Forest Cabin",
    image: "https://www.holidify.com/images/bgImages/JANAKPUR.jpg",
    rating: 4.3,
    description: "Cozy cabin surrounded by lush forest and hiking trails.",
    x: 30,
    y: 70,
    position: { lat: 27.7175, lng: 85.364 },
  },
];

export default function NearbyPlacesPage() {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const [selectedPlace, setSelectedPlace] = useState(null);

  const handleBackClick = () => {
    navigate(-1);
  };

  const handleLogout = () => {
    console.log("Logging out...");
    dispatch(logout());
    navigate("/login");
  };

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
        <h1 className="text-3xl font-bold text-gray-900 mb-6">
          Nearby Popular Places
        </h1>
        <div className="bg-white rounded-xl shadow-md overflow-hidden">
          <div className="flex flex-col lg:flex-row">
            <div className="lg:w-1/2 h-[400px] lg:h-auto">
              <Map famousPlaces={places} />
            </div>
            <div className="lg:w-1/2 p-6 overflow-y-auto max-h-[600px] lg:max-h-none">
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                Popular Places
              </h2>
              <div className="space-y-6">
                {places.map((place) => (
                  <div
                    key={place.id}
                    className={`flex items-start space-x-4 p-4 rounded-lg transition-colors duration-200 ${
                      selectedPlace?.id === place.id
                        ? "bg-indigo-50"
                        : "hover:bg-gray-50"
                    }`}
                    onMouseEnter={() => setSelectedPlace(place)}
                    onMouseLeave={() => setSelectedPlace(null)}
                  >
                    <img
                      src={place.image}
                      alt={place.name}
                      className="w-20 h-20 object-cover rounded-md"
                    />
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-gray-900">
                        {place.name}
                      </h3>
                      <p className="text-sm text-gray-500 mb-2">
                        {place.description}
                      </p>
                      <div className="flex items-center">
                        {[...Array(5)].map((_, i) => (
                          <Star
                            key={i}
                            className={`h-4 w-4 ${
                              i < Math.floor(place.rating)
                                ? "text-yellow-400 fill-current"
                                : "text-gray-300"
                            }`}
                          />
                        ))}
                        <span className="ml-2 text-sm text-gray-600">
                          {place.rating.toFixed(1)}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
