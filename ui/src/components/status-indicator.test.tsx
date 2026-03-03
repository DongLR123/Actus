import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { StatusIndicator } from "./status-indicator";

describe("StatusIndicator", () => {
  it("renders text and icon by meta", () => {
    render(
      <StatusIndicator
        meta={{ text: "执行中", tone: "warning", icon: "loader", spinning: true }}
      />
    );

    expect(screen.getByText("执行中")).toBeInTheDocument();
    expect(screen.getByTestId("status-icon-loader")).toBeInTheDocument();
  });
});
