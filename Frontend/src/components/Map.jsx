import { useEffect, useState } from "react";

const Map = () => {
  const [currentlatitude, setCurrentLatitude] = useState(27.671);
  const [currentlongitude, setCurrentLongitude] = useState(85.4291);
  const [destination, setDestination] = useState({
    lat: 27.671028,
    lng: 85.43925,
  });
  const [distance, setDistance] = useState(null);
  const [duration, setDuration] = useState(null);
  const [directionsService, setDirectionsService] = useState(null);
  const [directionsRenderer, setDirectionsRenderer] = useState(null);

  useEffect(() => {
    // Getting the user's current location
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const currentLat = position.coords.latitude;
          const currentLng = position.coords.longitude;

          // Update state with current location
          setCurrentLatitude(currentLat);
          setCurrentLongitude(currentLng);
        },
        (error) => {
          console.error("Error getting location: ", error);
          // Optionally, you can set a default location if there's an error
        }
      );
    } else {
      console.error("Geolocation is not supported by this browser.");
    }

    const map = new window.google.maps.Map(
      document.getElementById("map-container"),
      {
        center: { lat: currentlatitude, lng: currentlongitude },
        zoom: 12,
      }
    );

    const service = new window.google.maps.DirectionsService();

    const renderer = new window.google.maps.DirectionsRenderer({
      map: map,
      polylineOptions: {
        strokeColor: "#FF5733", // Line color
        strokeOpacity: 0.8, // Line opacity
        strokeWeight: 8, // Line thickness
      },
    });

    setDirectionsService(service);
    setDirectionsRenderer(renderer);

    // Custom marker
    // new window.google.maps.Marker({
    //   position: { lat: latitude, lng: longitude },
    //   map: map,
    //   animation: window.google.maps.Animation.DROP,
    // });

    // map.addListener("click", (event) => {
    //   const clickedLat = event.latLng.lat();
    //   const clickedLng = event.latLng.lng();

    //   setDestination({ lat: clickedLat, lng: clickedLng });

    //   // Ensure state updates before calling the functions
    //   setTimeout(() => {
    //     getRoute(latitude, longitude, clickedLat, clickedLng, service, renderer);
    //     getDistanceAndDuration(latitude, longitude, clickedLat, clickedLng);
    //   }, 100);
    // });

    getRoute(
      currentlatitude,
      currentlongitude,
      destination.lat,
      destination.lng,
      service,
      renderer
    );
    getDistanceAndDuration(
      currentlatitude,
      currentlongitude,
      destination.lat,
      destination.lng
    );
  }, [currentlatitude, currentlongitude, destination.lat, destination.lng]);

  // Function to draw the actual driving route
  function getRoute(lat1, lon1, lat2, lon2, service, renderer) {
    if (!service || !renderer) return;

    service.route(
      {
        origin: { lat: lat1, lng: lon1 },
        destination: { lat: lat2, lng: lon2 },
        travelMode: window.google.maps.TravelMode.DRIVING,
      },
      (result, status) => {
        if (status === "OK") {
          renderer.setDirections(result);
        } else {
          console.error("Error fetching directions:", status);
        }
      }
    );
  }

  // Function to get travel distance & duration
  function getDistanceAndDuration(lat1, lon1, lat2, lon2) {
    const service = new window.google.maps.DistanceMatrixService();
    service.getDistanceMatrix(
      {
        origins: [{ lat: lat1, lng: lon1 }],
        destinations: [{ lat: lat2, lng: lon2 }],
        travelMode: window.google.maps.TravelMode.DRIVING,
        unitSystem: window.google.maps.UnitSystem.METRIC,
      },
      (response, status) => {
        if (status === "OK") {
          const result = response.rows[0].elements[0];
          if (result.status === "OK") {
            setDistance(result.distance.text);
            setDuration(result.duration.text);
          } else {
            setDistance("Not Available");
            setDuration("Not Available");
          }
        } else {
          console.error("Distance Matrix API failed: ", status);
        }
      }
    );
  }

  const googleMapsUrl = `https://www.google.com/maps?q=${destination.lat},${destination.lng}`;
  return (
    <div>
      <div
        id="map-container"
        className="lg:w-full h-[400px] lg:h-[750px]"
      ></div>
      {destination && (
        <div className="mt-4 text-center">
          <p>
            📏 Distance: <strong>{distance}</strong>
          </p>
          <p>
            ⏳ Estimated Time: <strong>{duration}</strong>
          </p>
          <a
            href={googleMapsUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-4 text-blue-500 underline"
          >
            Go to Google Maps
          </a>
        </div>
      )}
    </div>
  );
};

export default Map;
