import { configureStore } from "@reduxjs/toolkit";
import counterReducer from "./slices/CounterSlice"
import authReducer from "./slices/AuthStoreSlice"

export const store = configureStore({
  reducer: {
    counter: counterReducer,
    auth: authReducer
  },
});

export default store;
