import { css } from "../../styled-system/css/css.mjs";

interface HeaderProps {
  faceDetectionEnabled: boolean;
  onFaceDetectionToggle: () => void;
}

export function Header({ faceDetectionEnabled, onFaceDetectionToggle }: HeaderProps) {
  return (
    <header
      className={css({
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      })}
    >
      <h1 className={css({ fontSize: "xl", fontWeight: "600" })}>
        Hierarchical Vision Pipeline
      </h1>

      <div className={css({ display: "flex", alignItems: "center", gap: "md" })}>
        <button
          onClick={onFaceDetectionToggle}
          className={css({
            paddingX: "lg",
            paddingY: "md",
            borderRadius: "lg",
            borderWidth: "2px",
            borderColor: faceDetectionEnabled ? "green.500" : "gray.300",
            backgroundColor: faceDetectionEnabled ? "green.50" : "white",
            color: faceDetectionEnabled ? "green.700" : "gray.700",
            _hover: {
              backgroundColor: faceDetectionEnabled ? "green.100" : "gray.100",
              borderColor: faceDetectionEnabled ? "green.600" : "gray.400"
            },
            fontWeight: "600",
            fontSize: "md",
            transition: "all 0.2s ease-in-out",
          })}
        >
          {faceDetectionEnabled ? "✓ Face Detection Enabled" : "Enable Face Detection"}
        </button>
      </div>
    </header>
  );
}