import { render, screen } from "@testing-library/react";
import App from "./App";

test("renders the SmartBidder dashboard header", () => {
  render(<App />);
  expect(screen.getByText(/SmartBidder/i)).toBeInTheDocument();
});
