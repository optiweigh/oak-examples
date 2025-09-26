import { css } from "../../styled-system/css/css.mjs";

export function ArrowDown({ label }: { label?: string }) {
  return (
    <div
      className={css({
        height: "48px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        position: "relative",
      })}
    >
      <svg width="24" height="36" viewBox="0 0 24 36" aria-hidden>
        <path d="M12 0 v26" stroke="#9CA3AF" strokeWidth="2" strokeLinecap="round" />
        <path d="M4 24 L12 34 L20 24" fill="none" stroke="#9CA3AF" strokeWidth="2" strokeLinecap="round" />
      </svg>
      {label ? (
        <span
          className={css({
            position: "absolute",
            top: "2",
            fontSize: "xs",
            color: "gray.600",
            textAlign: "center",
            px: "2",
            bg: "white",
            borderWidth: "1px",
            borderColor: "gray.200",
            borderRadius: "md",
          })}
          style={{ transform: "translateY(-100%)" }}
        >
          {label}
        </span>
      ) : null}
    </div>
  );
}
