import PropTypes from "prop-types";
import { useEffect, useState } from "react";

const Map = ({ famousPlaces }) => {
  const [locationName, setLocationName] = useState("");
  const [latitude, setLatitude] = useState(null);
  const [longitude, setLongitude] = useState(null);

  // const famousPlaces = [
  //   { name: "Kathmandu", position: { lat: 27.7172, lng: 85.324 } },
  //   { name: "Pokhara", position: { lat: 28.2096, lng: 83.9856 } },
  //   { name: "Lumbini", position: { lat: 27.6792, lng: 83.507 } },
  // ];

  const initializeMap = () => {
    const mapOptions = {
      center: { lat: 27.7172, lng: 85.324 }, // Default center
      zoom: 12, // Adjusting the zoom level as needed
    };

    const map = new window.google.maps.Map(
      document.getElementById("map-container"),
      mapOptions
    );

    famousPlaces.forEach((place) => {
      new window.google.maps.Marker({
        position: place.position,
        map: map,
        title: place.name,
      });
    });

    map.addListener("click", (event) => {
      const clickedLat = event.latLng.lat();
      const clickedLng = event.latLng.lng();

      setLatitude(clickedLat);
      setLongitude(clickedLng);

      const geocoder = new window.google.maps.Geocoder();
      geocoder.geocode(
        { location: { lat: clickedLat, lng: clickedLng } },
        (results, status) => {
          if (status === "OK" && results[0]) {
            const city = results[0].address_components[2].long_name;
            setLocationName(city);
          } else {
            console.error("Geocoding failed with status: ", status);
          }
        }
      );
      // marker.setPosition(event.latLng);
    });
  };

  useEffect(() => {
    initializeMap();
  }, []);

  return (
    <div>
      <div
        id="map-container"
        className="lg:w-full h-[400px] lg:h-[750px]"
      ></div>
    </div>
  );
};

Map.propTypes = {
  famousPlaces: PropTypes.arrayOf(
    PropTypes.shape({
      name: PropTypes.string,
      position: PropTypes.shape({
        lat: PropTypes.number,
        lng: PropTypes.number,
      }),
    })
  ),
};

export default Map;
