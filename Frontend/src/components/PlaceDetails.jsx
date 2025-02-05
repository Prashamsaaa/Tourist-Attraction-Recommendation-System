"use client";

import { useState } from "react";
import { Star, Calendar, Info, MapPin, ArrowLeft } from "lucide-react";
// import { MapContainer, Marker, Popup, TileLayer } from "react-leaflet";

import "leaflet/dist/leaflet.css";
import "leaflet-defaulticon-compatibility/dist/leaflet-defaulticon-compatibility.css";
import "leaflet-defaulticon-compatibility";
import { useLocation, useNavigate } from "react-router";
import Map from "./Map";

export default function PlaceDetails() {
  const location = useLocation();
  const { placeName } = location.state;

  const navigate = useNavigate();
  const [rating, setRating] = useState(0);

  const place = {
    name: "Bhaktapur Durbar Square",
    location: [27.671, 85.4276],
    address: "Durbar Square, Bhaktapur 44800, Nepal",
    description:
      "Bhaktapur Durbar Square is a historic palace complex and a UNESCO World Heritage site located in Bhaktapur, Nepal. It is known for its stunning architecture, intricate wood carvings, and ancient temples, reflecting the rich culture and history of the Malla dynasty.",
    itinerary: [
      "8:00 AM - Arrive at Bhaktapur Durbar Square",
      "8:30 AM - Explore the 55-Window Palace",
      "9:30 AM - Visit Vatsala Temple and Nyatapola Temple",
      "10:30 AM - Walk around the square and take in the ancient architecture",
      "12:00 PM - Enjoy lunch at a nearby local restaurant",
      "1:30 PM - Visit the Pottery Square and observe traditional pottery making",
    ],
    website: "https://www.welcomenepal.com/places-to-see/bhaktapur.html",
  };

  const handleBackClick = () => {
    navigate(-1);
  };

  return (
    <div className="container h-screen mx-auto p-4 space-y-6">
      <div className="mb-6 flex justify-between items-center">
        <button
          onClick={handleBackClick}
          className="flex items-center text-indigo-600 hover:text-indigo-800 transition-colors duration-200"
        >
          <ArrowLeft className="w-5 h-5 mr-2" />
          Back
        </button>
      </div>
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        <div className="p-6">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
            <div>
              <h1 className="text-4xl font-bold mb-2">{placeName}</h1>
              <p className="flex items-center text-sm text-gray-600">
                <MapPin className="w-4 h-4 mr-1" />
                {place.address}
              </p>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-right">
                <div className="text-3xl font-bold text-indigo-600 mb-1">
                  4.7
                </div>
                <div className="flex">
                  {[1, 2, 3, 4, 5].map((star) => (
                    <Star
                      key={star}
                      className="w-5 h-5 text-yellow-400 fill-current"
                    />
                  ))}
                </div>
                <div className="text-sm text-gray-600">(2,345 reviews)</div>
              </div>
              {/* <a
                href={place.website}
                target="_blank"
                rel="noopener noreferrer"
                className="hidden md:flex items-center px-4 py-2 border border-indigo-600 text-indigo-600 rounded-md hover:bg-indigo-600 hover:text-white transition-colors duration-200"
              >
                Visit Website
                <ExternalLink className="w-4 h-4 ml-2" />
              </a> */}
            </div>
          </div>
          <div className="grid lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-6">
              <div className="h-[400px] w-full rounded-lg overflow-hidden shadow-md">
                {/* <MapContainer
                  center={place.location}
                  zoom={15}
                  scrollWheelZoom={false}
                  style={{ height: "100%", width: "100%" }}
                >
                  <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  />
                  <Marker position={place.location}>
                    <Popup>{place.name}</Popup>
                  </Marker>
                </MapContainer> */}
                <Map />
              </div>
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-xl font-semibold flex items-center mb-4">
                  <Info className="w-5 h-5 mr-2" />
                  About
                </h2>
                <p className="text-gray-600">{place.description}</p>
              </div>
            </div>
            <div className="space-y-6">
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-xl font-semibold flex items-center mb-4">
                  <Calendar className="w-5 h-5 mr-2" />
                  Suggested Itinerary
                </h2>
                <ul className="space-y-3">
                  {place.itinerary.map((item, index) => (
                    <li key={index} className="flex items-start">
                      <span className="w-6 h-6 rounded-full bg-indigo-600 text-white text-xs flex items-center justify-center mr-3 mt-0.5 shrink-0">
                        {index + 1}
                      </span>
                      <span className="text-gray-600">{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="bg-indigo-600 text-white rounded-lg shadow-md p-6">
                <h2 className="text-xl font-semibold flex items-center mb-4">
                  <Star className="w-5 h-5 mr-2" />
                  Rate your experience
                </h2>
                <div className="flex flex-col items-center gap-4">
                  <div className="flex space-x-1">
                    {[1, 2, 3, 4, 5].map((star) => (
                      <button
                        key={star}
                        className={`p-1 rounded-full hover:bg-indigo-500 transition-colors duration-200 ${
                          star <= rating
                            ? "text-yellow-400"
                            : "text-white opacity-50"
                        }`}
                        onClick={() => setRating(star)}
                      >
                        <Star className="w-7 h-7 fill-current" />
                      </button>
                    ))}
                  </div>
                  <div className="text-4xl font-bold">
                    {rating > 0 ? rating.toFixed(1) : "-"}
                  </div>
                  {rating > 0 && (
                    <p className="text-center text-indigo-200">
                      Thanks for rating! Your feedback helps others.
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
