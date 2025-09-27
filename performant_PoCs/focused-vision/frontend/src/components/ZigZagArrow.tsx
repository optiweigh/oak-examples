import { css } from "../../styled-system/css/css.mjs";

interface ZigZagArrowProps {
  label: string;
  direction: "right" | "down-right" | "left";
}

export function ZigZagArrow({ label, direction }: ZigZagArrowProps) {
  const getArrowStyles = () => {
    switch (direction) {
      case "right":
        return {
          container: css({
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            paddingY: "md",
            position: "relative",
          }),
          arrow: css({
            width: "60px",
            height: "2px",
            backgroundColor: "#6b7280",
            position: "relative",
            "&::after": {
              content: '""',
              position: "absolute",
              right: "-6px",
              top: "-4px",
              width: "0",
              height: "0",
              borderLeft: "8px solid #6b7280",
              borderTop: "5px solid transparent",
              borderBottom: "5px solid transparent",
            },
          }),
          label: css({
            marginLeft: "md",
            fontSize: "sm",
            fontWeight: "600",
            color: "#6b7280",
            backgroundColor: "white",
            paddingX: "sm",
            paddingY: "xs",
            borderRadius: "md",
            border: "1px solid #e5e7eb",
          }),
        };
      case "down-right":
        return {
          container: css({
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            paddingY: "md",
            position: "relative",
            height: "100px",
          }),
          arrow: css({
            position: "relative",
            width: "120px",
            height: "80px",
            "&::before": {
              content: '""',
              position: "absolute",
              top: "0",
              left: "50%",
              width: "2px",
              height: "40px",
              backgroundColor: "#6b7280",
              transform: "translateX(-50%)",
            },
            "&::after": {
              content: '""',
              position: "absolute",
              bottom: "0",
              left: "50%",
              width: "80px",
              height: "2px",
              backgroundColor: "#6b7280",
              transform: "translateX(-50%)",
            },
          }),
          label: css({
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            fontSize: "sm",
            fontWeight: "600",
            color: "#6b7280",
            backgroundColor: "white",
            paddingX: "sm",
            paddingY: "xs",
            borderRadius: "md",
            border: "1px solid #e5e7eb",
            whiteSpace: "nowrap",
            zIndex: 10,
          }),
        };
      case "left":
        return {
          container: css({
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            paddingY: "md",
            position: "relative",
          }),
          arrow: css({
            width: "60px",
            height: "2px",
            backgroundColor: "#6b7280",
            position: "relative",
            "&::after": {
              content: '""',
              position: "absolute",
              left: "-6px",
              top: "-4px",
              width: "0",
              height: "0",
              borderRight: "8px solid #6b7280",
              borderTop: "5px solid transparent",
              borderBottom: "5px solid transparent",
            },
          }),
          label: css({
            marginRight: "md",
            fontSize: "sm",
            fontWeight: "600",
            color: "#6b7280",
            backgroundColor: "white",
            paddingX: "sm",
            paddingY: "xs",
            borderRadius: "md",
            border: "1px solid #e5e7eb",
          }),
        };
      default:
        return {};
    }
  };

  const styles = getArrowStyles();

  return (
    <div className={styles.container}>
      <div className={styles.arrow}></div>
      <div className={styles.label}>{label}</div>
      {direction === "down-right" && (
        <div className={css({
          position: "absolute",
          bottom: "0",
          right: "50%",
          transform: "translateX(40px)",
          width: "0",
          height: "0",
          borderLeft: "8px solid #6b7280",
          borderTop: "5px solid transparent",
          borderBottom: "5px solid transparent",
        })}></div>
      )}
    </div>
  );
}