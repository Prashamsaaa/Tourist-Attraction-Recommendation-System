import { createBrowserRouter, RouterProvider } from "react-router-dom"; // Import correct module
import LoginPage from "./components/Login";
import RegisterPage from "./components/Register";
import PreferencesPage from "./components/Preference";
import LandingPage from "./components/LandingPage";

// Custom error page component
function ErrorPage({ error }) {
  return (
    <div className="p-6 bg-red-100 text-red-800 border border-red-500 rounded-lg">
      <h1 className="text-2xl font-bold">Oops! Something went wrong.</h1>
      <p className="mt-4">Error: {error.message}</p>
    </div>
  );
}

const router = createBrowserRouter([
  {
    path: "/",
    element: <LandingPage />,
  
  },
  {
    path: "/login",
    element: <LoginPage />,
    errorElement: <ErrorPage />,  
  },
  {
    path: "/register",
    element: <RegisterPage />,
    errorElement: <ErrorPage />,
  },
  {
    path: "/preference",
    element: <PreferencesPage />,
    errorElement: <ErrorPage />,
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
