import { css } from "../../styled-system/css/css.mjs";

export function Header() {
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
        <div className={css({ fontSize: "sm", color: "gray.600" })}>
          Video → Face Mosaic → Eyes Mosaic
        </div>
      </div>
    </header>
  );
}