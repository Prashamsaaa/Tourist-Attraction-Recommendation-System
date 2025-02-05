import { createBrowserRouter, RouterProvider } from "react-router";

import { ToastContainer } from "react-toastify";
import "react-toastify/ReactToastify.css";

import LoginPage from "./components/Login";
import RegisterPage from "./components/Register";
import PreferencesPage from "./components/Preference";
import Places from "./components/NearbyPlaces";
import Map from "./components/Map";
import PlaceDetails from "./components/PlaceDetails";
import Index from "./components/LandingPage/Index";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Index />,
  },
  {
    name: "Login",
    path: "/login",
    element: <LoginPage />,
    errorElement: "Login Error here",
  },
  {
    path: "/register",
    element: <RegisterPage />,
    errorElement: "Register Error here",
  },
  {
    path: "/preference",
    element: <PreferencesPage />,
    errorElement: "Preference Error here",
  },
  {
    path: "/places",
    element: <Places />,
    errorElement: "Places Error here",
  },
  {
    path: "/details",
    element: <PlaceDetails />,
    errorElement: "No details found",
  },
  {
    path: "/map",
    element: <Map />,
    errorElement: "Map not found.",
  },
]);

function App() {
  return (
    <>
      <RouterProvider router={router} />
      <ToastContainer
        position="top-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick={false}
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="light"
      />
    </>
  );
}

export default App;
