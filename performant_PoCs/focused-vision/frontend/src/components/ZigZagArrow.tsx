import { css } from "../../styled-system/css/css.mjs";

interface ZigZagArrowProps {
  label: string;
  direction: "right" | "left";
}

export function ZigZagArrow({ label, direction }: ZigZagArrowProps) {
  if (direction === "right") {
    return (
      <div className={css({
        position: "absolute",
        top: "50%",
        left: "60%",
        transform: "translateY(-50%)",
        zIndex: 1000,
        pointerEvents: "none",
      })}>
        <div className={css({
          position: "relative",
          width: "120px",
          height: "150px",
        })}>
          {/* Horizontal line going right from tile edge */}
          <div className={css({
            position: "absolute",
            top: "0",
            left: "4px",
            width: "250px",
            height: "4px",
            backgroundColor: "#000000",
            borderRadius: "2px",
          })}></div>
          
          {/* Vertical line going down from turn point */}
          <div className={css({
            position: "absolute",
            top: "0",
            left: "250px",
            width: "4px",
            height: "330px",
            backgroundColor: "#000000",
            borderRadius: "2px",
          })}></div>
          
          {/* Single arrowhead pointing down at the end */}
          <div className={css({
            position: "absolute",
            top: "326px",
            left: "246px",
            width: "0",
            height: "0",
            borderTop: "12px solid #000000",
            borderLeft: "6px solid transparent",
            borderRight: "6px solid transparent",
          })}></div>
          
          {/* Label at the turn point */}
          <div className={css({
            position: "absolute",
            top: "10px",
            left: "259px",
            fontSize: "sm",
            fontWeight: "700",
            color: "white",
            fontFamily: "cursive",
            backgroundColor: "#000000",
            paddingX: "sm",
            paddingY: "xs",
            borderRadius: "md",
            whiteSpace: "nowrap",
            zIndex: 1001,
            border: "2px solid white",
          })}>
            {label}
          </div>
        </div>
      </div>
    );
  }

  if (direction === "left") {
    return (
      <div className={css({
        position: "absolute",
        top: "50%",
        right: "60%",
        transform: "translateY(-50%)",
        zIndex: 1000,
        pointerEvents: "none",
      })}>
        <div className={css({
          position: "relative",
          width: "120px",
          height: "150px",
        })}>
          {/* Horizontal line going left from tile edge */}
          <div className={css({
            position: "absolute",
            top: "0",
            right: "4px",
            width: "250px",
            height: "4px",
            backgroundColor: "#000000",
            borderRadius: "2px",
          })}></div>
          
          {/* Vertical line going down from turn point */}
          <div className={css({
            position: "absolute",
            top: "0",
            right: "250px",
            width: "4px",
            height: "330px",
            backgroundColor: "#000000",
            borderRadius: "2px",
          })}></div>
          
          {/* Single arrowhead pointing down at the end */}
          <div className={css({
            position: "absolute",
            top: "326px",
            right: "246px",
            width: "0",
            height: "0",
            borderTop: "12px solid #000000",
            borderLeft: "6px solid transparent",
            borderRight: "6px solid transparent",
          })}></div>
          
          {/* Label at the turn point */}
          <div className={css({
            position: "absolute",
            top: "10px",
            right: "259px",
            fontSize: "sm",
            fontWeight: "700",
            color: "white",
            fontFamily: "cursive",
            backgroundColor: "#000000",
            paddingX: "sm",
            paddingY: "xs",
            borderRadius: "md",
            whiteSpace: "nowrap",
            zIndex: 1001,
            border: "2px solid white",
          })}>
            {label}
          </div>
        </div>
      </div>
    );
  }

  return null;
}