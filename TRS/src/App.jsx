import { createBrowserRouter, RouterProvider } from "react-router";
import LoginPage from "./components/Login";
import RegisterPage from "./components/Register";
import PreferencesPage from "./components/Preference";

const router = createBrowserRouter([
  {
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
]);

function App() {
  return (
    <>
      <RouterProvider router={router} />
    </>
  );
}

export default App;
