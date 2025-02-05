import { Star } from "lucide-react";
import PropTypes from "prop-types";
import { useState } from "react";
import { useNavigate } from "react-router";

const DisplayCard = ({ userPreferences, places }) => {
  const navigate = useNavigate();

  // Group places by category
  const groupedPlaces = places.reduce((acc, place) => {
    if (!acc[place.category]) {
      acc[place.category] = [];
    }
    acc[place.category].push(place);
    return acc;
  }, {});

  const [ratings, setRatings] = useState(
    places.reduce((acc, place) => {
      acc[place.name] = place.rating; // Initialize with default rating
      return acc;
    }, {})
  );

  const handleRatingChange = (placeName, newRating) => {
    setRatings((prevRatings) => ({
      ...prevRatings,
      [placeName]: newRating,
    }));
    console.log(placeName, newRating);
  };

  const showRecommendedPlaces = (placeName, place) => {
    navigate("/details", { state: { placeName }});
  };

  return (
    <div>
      {userPreferences.length > 0 && (
        <div className="bg-white rounded-xl shadow-md overflow-hidden">
          <div className="p-8">
            {Object.keys(groupedPlaces).map((category) => (
              <div key={category} className="mb-8">
                <h2 className="text-2xl font-semibold text-gray-900 mb-6">
                  {category} Places
                </h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                  {groupedPlaces[category].map((place) => (
                    <div
                      key={place.name}
                      className="bg-white rounded-lg shadow-md overflow-hidden cursor-pointer"
                    >
                      <div
                        className="relative h-48"
                        onClick={()=>showRecommendedPlaces(place.name, place)}
                      >
                        <img
                          src={place.image}
                          alt={place.name}
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <div className="p-4">
                        <h3 className="text-lg font-semibold text-gray-900">
                          {place.name}
                        </h3>
                        <div className="flex items-center mt-2">
                          {[...Array(5)].map((_, i) => (
                            <Star
                              key={i}
                              className={`h-5 w-5 cursor-pointer ${
                                i < Math.floor(ratings[place.name])
                                  ? "text-yellow-400 fill-current"
                                  : "text-gray-300"
                              }`}
                              onClick={() =>
                                handleRatingChange(place.name, i + 1)
                              }
                            />
                          ))}
                          <span className="ml-2 text-sm text-gray-600">
                            {ratings[place.name].toFixed(1)}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Prop Validation
DisplayCard.propTypes = {
  userPreferences: PropTypes.arrayOf(PropTypes.string).isRequired,
  places: PropTypes.arrayOf(
    PropTypes.shape({
      name: PropTypes.string.isRequired,
      image: PropTypes.string.isRequired,
      rating: PropTypes.number.isRequired,
      category: PropTypes.string.isRequired,
    })
  ).isRequired,
};

export default DisplayCard;
