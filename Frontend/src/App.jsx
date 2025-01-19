import { createBrowserRouter, Navigate, RouterProvider } from "react-router";
import LoginPage from "./components/Login";
import RegisterPage from "./components/Register";
import PreferencesPage from "./components/Preference";
import Places from "./components/NearbyPlaces";
import Map from "./components/Map";
import Counter from "./components/Counter";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Navigate to={"login"} />,
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
    path: "/map",
    element: <Map />,
    errorElement: "Map not found.",
  },
  {
    path: "/counter",
    element: <Counter />,
    errorElement: "Error in counter"
  }
]);

function App() {
  return (
    <>
      <RouterProvider router={router} />
    </>
  );
}

export default App;
